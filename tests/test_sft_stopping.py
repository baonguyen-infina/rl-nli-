"""A full-budget NLI warm start must not terminate on answer-token loss."""
import sys
from types import ModuleType
import pytest
from train import sft


def test_full_budget_disables_the_early_stopping_callback(monkeypatch):
    fake=ModuleType('transformers')
    def unexpected(**kwargs):
        raise AssertionError('Disabled early stopping must not construct a callback')
    fake.EarlyStoppingCallback=unexpected
    monkeypatch.setitem(sys.modules,'transformers',fake)
    assert sft._sft_callbacks({'early_stopping_patience':None})==[]


def test_legacy_warm_start_keeps_patience_three(monkeypatch):
    fake=ModuleType('transformers')
    fake.EarlyStoppingCallback=lambda **kwargs:kwargs
    monkeypatch.setitem(sys.modules,'transformers',fake)
    assert sft._sft_callbacks({})==[{'early_stopping_patience':3}]


@pytest.mark.parametrize('patience',[False,0,-1,1.5,'3'])
def test_invalid_patience_is_rejected(patience):
    with pytest.raises(ValueError,match='patience'):
        sft._sft_callbacks({'early_stopping_patience':patience})
