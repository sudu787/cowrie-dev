# SPDX-FileCopyrightText: 2026 sudu787
#
# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Tests that the backend pool's guest list is changed under guest_lock
# ABOUTME: and that a consumer selects and claims a guest in one locked section.

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from backend_pool import util
from backend_pool.libvirt import backend_service
from backend_pool.pool_service import (
    POOL_STATE_AVAILABLE,
    POOL_STATE_DESTROYED,
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


if __name__ == "__main__":
    unittest.main()
