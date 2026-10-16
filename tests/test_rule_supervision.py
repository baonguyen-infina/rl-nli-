import pytest
from train.rule_supervision import rule_probabilities, alignment_status_positions


def test_single_alignment_line_keeps_three_status_probabilities():
    assert rule_probabilities([[.2,.5,.3]]) == pytest.approx([.2,.5,.3])


@pytest.mark.parametrize('rows,expected',[
    ([[1,0,0],[1,0,0]],[1,0,0]),
    ([[1,0,0],[0,1,0]],[0,1,0]),
    ([[0,1,0],[0,0,1]],[0,0,1]),
])
def test_rule_priority_matches_the_parser(rows,expected):
    assert rule_probabilities(rows)==pytest.approx(expected)


def test_mixed_status_distribution_sums_to_one():
    p=rule_probabilities([[.2,.5,.3],[.7,.1,.2]])
    assert sum(p)==pytest.approx(1)
    assert p==pytest.approx([.14,.42,.44])


def test_only_line_final_status_tokens_are_supervised():
    class Tok:
        pieces={1:'<align>\nH1: "h" <-> ∅ |',101:' missing',2:'\n</align><answer>neutral</answer>'}
        def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)
    assert alignment_status_positions([1,101,2],Tok(),[100,101,102])==[1]


def test_rule_loss_uses_the_predictive_position_and_has_gold_gradient():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from train.rule_supervision import rule_supervision_loss
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores=torch.nn.Parameter(torch.zeros(2,6,8))
        def forward(self,input_ids,attention_mask,logits_to_keep,**kwargs):
            assert input_ids.shape==(2,6) and attention_mask.shape==(2,6)
            assert logits_to_keep.tolist()==[3,4]
            return SimpleNamespace(logits=self.scores[:,logits_to_keep,:])
    model=Model()
    inputs={'prompt_ids':torch.ones(2,3,dtype=torch.long),'prompt_mask':torch.ones(2,3),
            'completion_ids':torch.ones(2,3,dtype=torch.long),'completion_mask':torch.ones(2,3),
            'rule_positions':torch.tensor([[1,-1],[2,-1]]),'rule_gold':torch.tensor([1,2])}
    loss,fraction=rule_supervision_loss(model,inputs,[1,2,3])
    assert fraction==1 and loss.item()==pytest.approx(1.0986123)
    loss.backward()
    assert model.scores.grad[0,3,2]<0 and model.scores.grad[1,4,3]<0
    assert model.scores.grad[0,4].abs().sum()==0


def test_schema_support_preserves_empty_reference_binding():
    from formats.align import AlignLine
    from train.rule_supervision import schema_status_support
    assert schema_status_support([AlignLine(1,'h','p','match'),AlignLine(2,'h',None,'missing')])==[[True,False,True],[False,True,False]]


def test_schema_loss_does_not_push_missing_on_a_quoted_premise():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from train.rule_supervision import rule_supervision_loss
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scores=torch.nn.Parameter(torch.zeros(1,3,4))
        def forward(self,logits_to_keep,**kwargs):return SimpleNamespace(logits=self.scores[:,logits_to_keep])
    model=Model()
    inputs={'prompt_ids':torch.ones(1,1,dtype=torch.long),'prompt_mask':torch.ones(1,1),
            'completion_ids':torch.ones(1,2,dtype=torch.long),'completion_mask':torch.ones(1,2),
            'rule_positions':torch.tensor([[1]]),'rule_gold':torch.tensor([0]),
            'rule_support':torch.tensor([[[True,False,True]]])}
    loss,fraction=rule_supervision_loss(model,inputs,[1,2,3])
    assert fraction==1 and loss.item()==pytest.approx(.6931472)
    loss.backward()
    assert model.scores.grad[0,1,2]==0
    assert model.scores.grad[0,1,1]<0
    # No feasible neutral status assignment exists with only quoted premise lines.
    inputs['rule_gold']=torch.tensor([1])
    loss,fraction=rule_supervision_loss(model,inputs,[1,2,3])
    assert loss is None and fraction==0
