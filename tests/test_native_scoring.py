"""Detached assignment scores must repeat exactly and leave training intact."""
import pytest
import torch
from torch import nn
import train.native_training as nt


def test_assignment_scores_repeat_without_dropout_or_gradients():
    torch.manual_seed(19)
    model=nn.Sequential(nn.Linear(8,8),nn.Dropout(0.8),nn.Linear(8,3)).train()
    features=torch.ones(4,8,requires_grad=True)
    with nt.deterministic_scoring(model):
        scores=[model(features) for _ in range(4)]
        assert all(torch.equal(scores[0],value) for value in scores[1:])
        assert not any(value.requires_grad for value in scores)
        assert not torch.is_grad_enabled()
    assert model.training and torch.is_grad_enabled()
    model(features).sum().backward()
    assert model[0].weight.grad is not None
    assert model[0].weight.grad.abs().sum()>0


def test_scoring_restores_mixed_module_modes_after_error():
    model=nn.Sequential(nn.Linear(2,2),nn.Dropout(0.5)).train()
    model[0].eval()
    before={name:module.training for name,module in model.named_modules()}
    with pytest.raises(RuntimeError,match='scoring failed'):
        with nt.deterministic_scoring(model):
            assert all(not module.training for module in model.modules())
            raise RuntimeError('scoring failed')
    assert {name:module.training for name,module in model.named_modules()}==before
    assert torch.is_grad_enabled()


def test_eval_model_stays_in_eval_after_scoring():
    model=nn.Sequential(nn.Linear(2,2),nn.Dropout(0.5)).eval()
    with nt.deterministic_scoring(model):
        model(torch.ones(1,2))
    assert all(not module.training for module in model.modules())


def test_scoring_preserves_an_outer_no_grad_context():
    model=nn.Linear(2,2).train()
    with torch.no_grad():
        with nt.deterministic_scoring(model):
            model(torch.ones(1,2))
        assert not torch.is_grad_enabled()
    assert model.training and torch.is_grad_enabled()
