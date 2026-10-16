"""Remote CPU checks for native attribution aggregation and token credit."""
import pytest


@pytest.fixture
def torch():return pytest.importorskip('torch')


def helper():
    from train.native_saliency import native_token_credit
    return native_token_credit


def test_label_vectors_cancel_before_magnitude(torch):
    value=torch.tensor([[[[2.,-3.],[1.,0.]], [[-2.,3.],[-1.,0.]], [[0.,0.],[0.,0.]]]])
    credit=helper()(value,[2],[1],floor=.25)
    assert credit.tolist()==[[1.,1.]]


def test_credit_matches_hand_calculation_and_has_mean_one(torch):
    value=torch.zeros(1,3,3,2)
    value[0,0,:,0]=torch.tensor([1.,2.,3.])
    credit=helper()(value,[3],[1],floor=.25)
    assert credit[0].tolist()==pytest.approx([.625,1.,1.375])
    assert credit.sum().item()==pytest.approx(3.)
    assert not credit.requires_grad


def test_padding_and_unused_positions_are_zero_even_if_nonfinite(torch):
    value=torch.zeros(2,3,4,2)
    value[0,:,2:]=float('nan');value[1]=float('inf')
    credit=helper()(value,[2,4],[1,0],floor=.25)
    assert credit.tolist()==[[1.,1.,0.,0.],[0.,0.,0.,0.]]


def test_zero_length_and_no_contrast_use_valid_uniform_fallback(torch):
    value=torch.zeros(3,3,4,2)
    credit=helper()(value,[0,3,1],[1,1,1],floor=.25)
    assert credit.tolist()==[[0.,0.,0.,0.],[1.,1.,1.,0.],[1.,0.,0.,0.]]


def test_credit_is_invariant_to_positive_attribution_scale_and_label_order(torch):
    value=torch.arange(2*3*4*2,dtype=torch.float32).reshape(2,3,4,2).sin()
    a=helper()(value,[4,2],[1,1],floor=.25)
    b=helper()(value[:,[2,0,1]]*7,[4,2],[1,1],floor=.25)
    assert torch.allclose(a,b,atol=1e-6)


def test_detached_credit_does_not_backpropagate_through_estimator(torch):
    value=torch.ones(1,3,4,2,requires_grad=True)
    credit=helper()(value,[4],[1],floor=.25)
    assert credit.grad_fn is None and not credit.requires_grad
    assert value.grad is None


@pytest.mark.parametrize('floor',[-.1,1.1,float('nan')])
def test_invalid_floor_rejected(torch,floor):
    with pytest.raises(ValueError):helper()(torch.zeros(1,3,4,2),[4],[1],floor=floor)


@pytest.mark.parametrize('lengths,active',[([5],[1]),([-1],[1]),([4],[.5]),([],[1])])
def test_invalid_lengths_and_weights_rejected(torch,lengths,active):
    with pytest.raises(ValueError):helper()(torch.zeros(1,3,4,2),lengths,active,floor=.25)


def test_active_nonfinite_and_wrong_label_shape_rejected(torch):
    for value in (torch.ones(1,2,4,2),torch.full((1,3,4,2),float('nan'))):
        with pytest.raises(ValueError):helper()(value,[4],[1],floor=.25)


def test_extreme_finite_attribution_does_not_overflow(torch):
    value=torch.full((1,3,4,8),1e38)
    credit=helper()(value,[4],[1],floor=.25)
    assert torch.isfinite(credit).all() and credit.tolist()==[[1.,1.,1.,1.]]


def test_reasoning_credit_only_weights_sampled_reasoning_positions(torch):
    from train.native_saliency import apply_reasoning_credit
    advantages=torch.tensor([2.,-1.])
    credit=torch.tensor([[.5,1.5,0.],[1.,0.,0.]])
    weighted=apply_reasoning_credit(advantages,credit,[2,1],opener_length=1,completion_length=7)
    assert weighted.tolist()==[[2.,1.,3.,2.,2.,2.,2.],[-1.,-1.,-1.,-1.,-1.,-1.,-1.]]


def test_reasoning_credit_cannot_overrun_completion(torch):
    from train.native_saliency import apply_reasoning_credit
    with pytest.raises(ValueError):
        apply_reasoning_credit(torch.ones(1),torch.ones(1,4),[4],opener_length=1,completion_length=3)
