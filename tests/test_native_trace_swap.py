"""Native trace rotation never receives labels or prompt metadata."""
import pytest
from train.native_trace_swap import native_swap_indices


def test_rotate_within_each_complete_rollout_group():
    members={(7,0):[0,2,4],(7,1):[1,3,5]}
    assert native_swap_indices(members,[1]*6)==[2,3,4,5,0,1]


def test_padding_never_donates_or_receives_a_rotated_prefix():
    members={(7,0):[0,2,4],(7,1):[1,3,5]}
    assert native_swap_indices(members,[1,1,0,0,1,1])==[4,5,2,3,0,1]


def test_singleton_and_empty_active_groups_preserve_prefixes():
    assert native_swap_indices({0:[0,1,2],1:[3,4,5]},[0,1,0,0,0,0])==list(range(6))
    assert native_swap_indices({},[])==[]


def test_rotation_has_no_cross_group_or_duplicate_donors():
    members={0:[4,0,2],1:[5,1,3]}
    mapping=native_swap_indices(members,[1,0,1,1,1,1])
    for indices in members.values():
        active=[i for i in indices if [1,0,1,1,1,1][i]]
        assert sorted(mapping[i] for i in active)==sorted(active)
        assert all(mapping[i]!=i for i in active)


@pytest.mark.parametrize('members,weights',[
    ({0:[0]},[.5]),({0:[0]},[1,1]),({0:[0,0]},[1]),
    ({0:[0],1:[0]},[1]),({0:[-1]},[1]),({0:[1]},[1]),
    ({0:[]},[]),({0:[0,1,2,3]},[1]*4),({0:[0.,1]},[1,1]),
])
def test_invalid_membership_or_weights_raise(members,weights):
    with pytest.raises(ValueError):native_swap_indices(members,weights)
