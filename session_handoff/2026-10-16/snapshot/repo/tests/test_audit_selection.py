import pytest
from analysis.audit_rollouts import select_rows


def test_latest_unique_late_prompts_are_sampled_reproducibly():
    rows=[{'id':str(i),'step':step,'completion':f'{i}-{step}'} for step in [1,150,151] for i in range(60)]
    a=select_rows(rows)
    assert len(a)==50 and len({r['id'] for r in a})==50
    assert all(r['step']==151 for r in a)
    assert a==select_rows(rows)


def test_short_audit_is_not_silently_reported_as_fifty():
    with pytest.raises(ValueError):select_rows([{'id':'x','step':151}])
