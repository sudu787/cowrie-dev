# SPDX-FileCopyrightText: 2026 sudu787
#
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Tests the backend pool's locking of its guest list and that every
# ABOUTME: producer iteration runs in a worker thread, off the reactor.

from __future__ import annotations

import sys
import unittest
from typing import Any
from unittest import mock

from twisted.internet import defer, task, threads

from backend_pool import pool_service, util
from backend_pool.libvirt import backend_service
from backend_pool.pool_service import (
    POOL_STATE_AVAILABLE,
    POOL_STATE_DESTROYED,
    POOL_STATE_UNAVAILABLE,
    POOL_STATE_USED,
    POOL_STATE_USING,
    Guest,
    NoAvailableVMs,
    PoolService,
)
from cowrie.test.configoverride import override_option


class LockCheckedList(list):
    """A guests list that records whether guest_lock was held on each change."""

    def __init__(self, service: PoolService) -> None:
        super().__init__()
        self.service = service
        self.unlocked_changes: list[str] = []

    def _check(self, op: str) -> None:
        if not self.service.guest_lock.locked():
            self.unlocked_changes.append(op)

    def append(self, item: Any) -> None:
        self._check("append")
        super().append(item)

    def remove(self, item: Any) -> None:
        self._check("remove")
        super().remove(item)


def make_guest(guest_id: int, state: str, **kw: Any) -> Guest:
    fields: dict[str, Any] = {
        "id": guest_id,
        "client_ips": [],
        "connected": 0,
        "state": state,
        "prev_state": None,
        "start_timestamp": 0.0,
        "freed_timestamp": -1,
        "domain": f"dom{guest_id}",
        "guest_ip": f"192.168.150.{guest_id + 10}",
        "name": "",
        "snapshot": f"snap{guest_id}",
    }
    fields.update(kw)
    return Guest(**fields)


def make_service(test: unittest.TestCase) -> PoolService:
    """A PoolService with libvirt and nmap replaced by mocks."""
    override_option(test, "backend_pool", "guest_ssh_port", "22")
    backend = mock.patch.object(backend_service, "LibvirtBackendService")
    backend.start()
    test.addCleanup(backend.stop)
    nmap = mock.patch.object(util, "nmap_port", return_value=True)
    nmap.start()
    test.addCleanup(nmap.stop)
    return PoolService(mock.MagicMock())


class GuestListLockingTests(unittest.TestCase):
    """The producer changes the guests list only while holding guest_lock
    (issue #40453)."""

    def setUp(self) -> None:
        self.service = make_service(self)
        self.guests = LockCheckedList(self.service)
        self.service.guests = self.guests

    def test_new_guests_are_added_under_the_lock(self) -> None:
        held_during_create: list[bool] = []

        def create_guest(is_ip_free: Any) -> tuple[str, str, str]:
            held_during_create.append(self.service.guest_lock.locked())
            return ("dom", "snap", f"192.168.150.{len(held_during_create) + 10}")

        self.service.qemu.create_guest.side_effect = create_guest
        self.service.max_vm = 2
        self.service._PoolService__producer_create_guests()  # type: ignore[attr-defined]

        self.assertEqual(len(self.guests), 2)
        self.assertEqual(self.guests.unlocked_changes, [])
        # The slow libvirt call must not hold up consumers.
        self.assertEqual(held_during_create, [False, False])

    def test_destroyed_guests_are_removed_under_the_lock(self) -> None:
        list.extend(
            self.guests,
            [make_guest(0, POOL_STATE_DESTROYED), make_guest(1, POOL_STATE_USED)],
        )
        self.service._PoolService__producer_remove_destroyed()  # type: ignore[attr-defined]

        self.assertEqual([g.id for g in self.guests], [1])
        self.assertEqual(self.guests.unlocked_changes, [])


class RequestVmTests(unittest.TestCase):
    """request_vm selects and claims a guest in one locked section."""

    def setUp(self) -> None:
        self.service = make_service(self)

    def test_guest_is_claimed_while_the_lock_is_held(self) -> None:
        held_on_claim: list[bool] = []
        service = self.service

        class WatchedGuest(Guest):
            def __setattr__(self, name: str, value: Any) -> None:
                if name == "state" and value == POOL_STATE_USING:
                    held_on_claim.append(service.guest_lock.locked())
                super().__setattr__(name, value)

        guest = make_guest(3, POOL_STATE_AVAILABLE)
        guest.__class__ = WatchedGuest
        service.guests = [guest]

        self.assertEqual(
            service.request_vm("203.0.113.5"), (3, "192.168.150.13", "snap3")
        )
        self.assertEqual(held_on_claim, [True])
        self.assertFalse(service.guest_lock.locked())

    def test_same_client_gets_its_guest_back(self) -> None:
        mine = make_guest(1, POOL_STATE_USED, client_ips=["203.0.113.5"])
        other = make_guest(2, POOL_STATE_AVAILABLE)
        self.service.guests = [other, mine]

        guest_id, _, _ = self.service.request_vm("203.0.113.5")

        self.assertEqual(guest_id, 1)
        self.assertEqual(mine.state, POOL_STATE_USING)
        self.assertEqual(mine.prev_state, POOL_STATE_USED)
        self.assertEqual(mine.connected, 1)

    def test_free_and_reuse_still_update_the_guest(self) -> None:
        guest = make_guest(4, POOL_STATE_AVAILABLE)
        self.service.guests = [guest]
        self.service.request_vm("203.0.113.5")
        self.service.reuse_vm(4)
        self.assertEqual((guest.state, guest.connected), (POOL_STATE_AVAILABLE, 0))
        self.service.request_vm("203.0.113.5")
        self.service.free_vm(4)
        self.assertEqual((guest.state, guest.connected), (POOL_STATE_USED, 0))

    def test_no_guest_raises_without_holding_the_lock(self) -> None:
        self.service.guests = []
        with self.assertRaises(NoAvailableVMs):
            self.service.request_vm("203.0.113.5")
        self.assertFalse(self.service.guest_lock.locked())


class ProducerLoopTests(unittest.TestCase):
    """Every producer iteration runs in a worker thread, and stopping or
    restarting the pool stops the old loop (issue #40449)."""

    def setUp(self) -> None:
        # stop_pool() imports libvirt lazily; it is not installed in tests.
        libvirt: Any = mock.MagicMock()
        libvirt.libvirtError = type("libvirtError", (Exception,), {})
        modules = mock.patch.dict(sys.modules, {"libvirt": libvirt})
        modules.start()
        self.addCleanup(modules.stop)

        self.service = make_service(self)
        self.service.qemu.create_guest.return_value = None
        self.clock = task.Clock()
        for target, attr, value in (
            (pool_service, "reactor", self.clock),
            (threads, "deferToThread", self.defer_to_thread),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.handoffs: list[tuple[Any, defer.Deferred[None]]] = []

    def defer_to_thread(self, f: Any) -> defer.Deferred[None]:
        d: defer.Deferred[None] = defer.Deferred()
        self.handoffs.append((f, d))
        return d

    def finish_iteration(self, index: int = -1) -> None:
        self.handoffs[index][1].callback(None)

    def producer_timers(self) -> list[Any]:
        # start_pool() may also arm the separate recycle timer
        # ([backend_pool] recycle_period); only producer timers matter here.
        return [
            c
            for c in self.clock.getDelayedCalls()
            if getattr(c, "func", None) == self.service._run_producer
        ]

    def test_every_iteration_is_handed_to_a_thread(self) -> None:
        self.service.start_pool()
        for _ in range(3):
            self.finish_iteration()
            self.clock.advance(self.service.loop_sleep_time)
        self.assertEqual(len(self.handoffs), 4)
        for f, _ in self.handoffs:
            self.assertEqual(f, self.service.producer_loop)

    def test_producer_loop_schedules_nothing_itself(self) -> None:
        # It runs in a worker thread, where touching the reactor is unsafe.
        self.service.producer_loop()
        self.assertEqual(self.producer_timers(), [])

    def test_next_iteration_waits_for_the_previous_one(self) -> None:
        self.service.start_pool()
        self.clock.advance(60)
        self.assertEqual(len(self.handoffs), 1)
        self.finish_iteration()
        self.clock.advance(self.service.loop_sleep_time)
        self.assertEqual(len(self.handoffs), 2)

    def test_failed_iteration_is_logged_and_the_loop_goes_on(self) -> None:
        with mock.patch.object(PoolService, "_log") as log:
            self.service.start_pool()
            self.handoffs[0][1].errback(RuntimeError("libvirt went away"))
            self.clock.advance(self.service.loop_sleep_time)
        log.failure.assert_called_once()
        self.assertEqual(len(self.handoffs), 2)

    def test_stop_during_an_iteration(self) -> None:
        self.service.start_pool()
        self.service.stop_pool()  # must not raise AlreadyCalled
        self.finish_iteration()
        self.assertEqual(self.producer_timers(), [])

    def test_stop_between_iterations_cancels_the_timer(self) -> None:
        self.service.start_pool()
        self.finish_iteration()
        self.service.stop_pool()
        self.assertEqual(self.producer_timers(), [])
        self.service.stop_pool()  # a second stop must not raise either

    def test_restart_during_an_iteration_runs_one_loop(self) -> None:
        self.service.start_pool()
        self.service.restart_pool()
        self.assertEqual(len(self.handoffs), 2)
        self.finish_iteration(0)  # the old loop's iteration
        self.assertEqual(self.producer_timers(), [])
        self.finish_iteration(1)
        self.assertEqual(len(self.producer_timers()), 1)


class HealthCheckTests(unittest.TestCase):
    """The health check probes guests without holding guest_lock."""

    def setUp(self) -> None:
        self.service = make_service(self)

    def check_health(self) -> None:
        self.service._PoolService__producer_check_health()  # type: ignore[attr-defined]

    def test_probes_run_without_the_lock(self) -> None:
        held: list[bool] = []

        def probe(ip: str, port: int) -> bool:
            held.append(self.service.guest_lock.locked())
            return ip != "192.168.150.11"

        self.service.guests = [
            make_guest(1, POOL_STATE_AVAILABLE),
            make_guest(2, POOL_STATE_USED),
        ]
        with mock.patch.object(util, "nmap_port", side_effect=probe):
            self.check_health()
        self.assertTrue(held)
        self.assertFalse(any(held))
        states = [g.state for g in self.service.guests]
        self.assertEqual(states, [POOL_STATE_UNAVAILABLE, POOL_STATE_USED])

    def test_guest_that_changed_during_the_probe_is_left_alone(self) -> None:
        guest = make_guest(1, POOL_STATE_USED)
        self.service.guests = [guest]

        def probe(ip: str, port: int) -> bool:
            guest.state = POOL_STATE_DESTROYED
            return False

        with mock.patch.object(util, "nmap_port", side_effect=probe):
            self.check_health()
        self.assertEqual(guest.state, POOL_STATE_DESTROYED)


if __name__ == "__main__":
    unittest.main()
