"""Synthetic remote tests for finite-label reasoning utility; no benchmark data."""
import pytest
from common.labels import LABELS


def macro_f1(gold, predictions):
    values=[]
    for c in range(3):
        tp=sum(y==c and p==c for y,p in zip(gold,predictions))
        denom=sum(y==c for y in gold)+sum(p==c for p in predictions)
        values.append(2*tp/denom if denom else 0.)
    return sum(values)/3


def test_counterfactual_f1_matches_reconstructing_each_candidate():
    from train.label_marginal import counterfactual_utilities
    gold=[0,1,2,0,1,2];pred=[0,2,1,2,1,0];rollouts=[0,0,0,1,1,1]
    got=counterfactual_utilities(gold,pred,[1]*6,rollouts)
    for i in range(6):
        other=[j for j in range(6) if rollouts[j]==rollouts[i] and j!=i]
        baseline=macro_f1([gold[j] for j in other],[pred[j] for j in other])
        for c in range(3):
            value=3*(macro_f1([gold[j] for j in other]+[gold[i]],
                              [pred[j] for j in other]+[c])-baseline)
            assert got[i][c]==pytest.approx(value)


def test_own_sampled_answer_does_not_change_own_utility():
    from train.label_marginal import counterfactual_utilities
    a=counterfactual_utilities([0,1,2],[0,0,2],[1,1,1],[0,0,0])
    b=counterfactual_utilities([0,1,2],[2,0,2],[1,1,1],[0,0,0])
    assert a[0]==pytest.approx(b[0])


def test_padding_does_not_contribute_to_confusion_counts():
    from train.label_marginal import counterfactual_utilities
    a=counterfactual_utilities([0,1],[1,1],[1,1],[0,0])
    b=counterfactual_utilities([0,1,-1],[1,1,-1],[1,1,0],[0,0,0])
    assert a==b[:2] and b[2]==[0.,0.,0.]


def test_rloo_uses_examples_and_excludes_inactive_rows():
    from train.label_marginal import example_rloo
    got=example_rloo([1.,3.,5.,7.,float('nan')],['a','a','b','b','padding'],[1,1,1,1,0])
    assert got==pytest.approx([-2.,2.,-2.,2.,0.])


@pytest.mark.parametrize('objective',['accuracy','macro_f1'])
def test_linearization_replays_value_and_gradient_across_shuffled_slices(objective):
    torch=pytest.importorskip('torch')
    from train.label_marginal import label_marginal_linearization
    from train.native_allocation import allocation_surrogate
    scores=torch.tensor([[1.,.5,-.2],[.2,1.1,.4],[.8,.1,1.3],
                         [.3,.9,.2],[1.2,.1,.5],[.2,.4,.8]],dtype=torch.float64)
    labels=[LABELS[i] for i in (0,0,1,1,2,2)]
    cache=label_marginal_linearization(scores,labels,[1]*6,[0,0,1,1,2,2],[0,1,0,1,0,1],
                                      objective=objective,seed=11)
    live=scores.clone().requires_grad_()
    expected=-(live.softmax(-1)*cache['utilities']).sum()/6
    expected_grad=torch.autograd.grad(expected,live)[0]
    replay=allocation_surrogate(live,cache['scores'],cache['jacobian'],cache['value_share'],cache['group_count'])
    assert replay.item()==pytest.approx(expected.item())
    assert torch.allclose(torch.autograd.grad(replay,live)[0],expected_grad,atol=1e-10)
    pieces=live.sum()*0
    for idx in ([5,1],[3,0],[4,2]):
        pieces=pieces+allocation_surrogate(live[idx],cache['scores'][idx],cache['jacobian'][idx],
                                           cache['value_share'][idx],cache['group_count'])
    assert pieces.item()==pytest.approx(expected.item())
    assert torch.allclose(torch.autograd.grad(pieces,live)[0],expected_grad,atol=1e-10)
    assert not any(cache[k].requires_grad for k in ('scores','jacobian','value_share','utilities','rewards'))


def test_accuracy_reward_is_gold_probability_even_if_sampled_answers_all_wrong():
    torch=pytest.importorskip('torch')
    from train.label_marginal import label_marginal_linearization
    scores=torch.tensor([[0.,1.,2.],[1.,.5,0.]])
    cache=label_marginal_linearization(scores,[LABELS[0]]*2,[1,1],['a','a'],[0,1],
                                      objective='accuracy',seed=0)
    assert torch.allclose(cache['rewards'],scores.softmax(-1)[:,0])
    assert cache['jacobian'].abs().sum()>0


def test_sampling_is_reproducible_and_inactive_nonfinite_scores_are_ignored():
    torch=pytest.importorskip('torch')
    from train.label_marginal import label_marginal_linearization
    scores=torch.tensor([[1.,2.,3.],[2.,1.,3.],[float('nan')]*3])
    args=(scores,[LABELS[0],LABELS[0],'ignored'],[1,1,0],['a','a','pad'],[0,1,0])
    a=label_marginal_linearization(*args,objective='macro_f1',seed=42)
    b=label_marginal_linearization(*args,objective='macro_f1',seed=42)
    assert torch.equal(a['sampled_labels'],b['sampled_labels'])
    assert torch.isfinite(a['scores']).all() and torch.isfinite(a['jacobian']).all()
    assert a['rewards'][2]==0 and a['jacobian'][2].abs().sum()==0


def test_active_examples_need_complete_unique_rollout_slots():
    torch=pytest.importorskip('torch')
    from train.label_marginal import label_marginal_linearization
    with pytest.raises(ValueError,match='rollout'):
        label_marginal_linearization(torch.zeros(2,3),[LABELS[0]]*2,[1,1],['a','a'],[0,0],
                                     objective='accuracy',seed=0)
