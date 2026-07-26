"""M5 MultiplexStateV1 assignment and tensor boundary tests."""

from __future__ import annotations

import pytest
import torch

from sam3.export import Demux, Mux, ScatterReplace
from sam3.runtime.multiplex_state import (
    BUCKET_CAPACITY,
    MultiplexCapacityError,
    MultiplexStateV1,
)


@pytest.mark.parametrize("active", [1, 2, 15, 16])
def test_mux_demux_identity_and_zero_padding(active: int) -> None:
    state = MultiplexStateV1()
    for object_id in range(active):
        state.add_object(object_id)
    assignments = torch.from_numpy(state.assignment_array())
    validity = torch.from_numpy(state.validity_array(bucket_count=1))
    values = torch.arange(active * 6, dtype=torch.float32).view(active, 2, 3)
    bucket = Mux(1)(values, assignments, validity)
    assert torch.equal(Demux(1)(bucket, assignments, validity), values)
    assert torch.count_nonzero(bucket[:, active:]) == 0


def test_empty_and_removed_slots_do_not_affect_active_values() -> None:
    state = MultiplexStateV1()
    for object_id in (10, 20, 30):
        state.add_object(object_id)
    removed = state.remove_object(20)
    values = torch.tensor([[1.0], [3.0]])
    assignments = torch.from_numpy(state.assignment_array([10, 30]))
    validity = torch.from_numpy(state.validity_array(bucket_count=1))
    bucket = Mux(1)(values, assignments, validity)
    bucket[removed.bucket, removed.slot] = 999.0
    assert Demux(1)(bucket, assignments, validity).flatten().tolist() == [1.0, 3.0]


def test_add_remove_readd_replace_and_revision() -> None:
    state = MultiplexStateV1()
    first = state.add_object(7)
    second = state.add_object(8)
    assert (first.bucket, first.slot) == (0, 0)
    assert (second.bucket, second.slot) == (0, 1)
    assert state.revision == 2
    removed = state.remove_object(7)
    assert removed == first
    reused = state.add_object(9)
    assert reused == first
    assert state.replace_object(9, 11) == first
    assert state.object_ids == (8, 11)
    assert state.revision == 5


def test_second_bucket_capacity_and_no_compaction() -> None:
    state = MultiplexStateV1()
    for object_id in range(32):
        state.add_object(object_id)
    assert state.require_object(15).bucket == 0
    assert state.require_object(16).bucket == 1
    with pytest.raises(MultiplexCapacityError, match="32"):
        state.add_object(32)
    state.remove_object(0)
    assert state.require_object(16).slot == 0
    assert state.add_object(100).bucket == 0


def test_scatter_replace_preserves_non_target_slots() -> None:
    values = torch.arange(2 * BUCKET_CAPACITY * 4, dtype=torch.float32).view(
        2, BUCKET_CAPACITY, 4
    )
    before = values.clone()
    assignment = torch.tensor([[1, 0]], dtype=torch.int64)
    replacement = torch.full((1, 4), -1.0)
    output = ScatterReplace(2)(values, replacement, assignment)
    assert torch.equal(output[1, 0], replacement[0])
    before[1, 0] = replacement[0]
    assert torch.equal(output, before)


def test_invalid_assignment_is_rejected() -> None:
    values = torch.ones((1, 1))
    validity = torch.ones((1, BUCKET_CAPACITY), dtype=torch.bool)
    with pytest.raises(Exception, match="unique"):
        Mux(1)(values.expand(2, 1), torch.tensor([[0, 0], [0, 0]]), validity)
