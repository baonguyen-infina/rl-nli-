"""Exercise real native reward/loss hooks with a tiny model on remote CPU."""
from collections import defaultdict
from types import SimpleNamespace
import pytest


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


class Tokenizer:
    pad_token_id=0
    eos_token_id=9
    def encode(self,text,add_special_tokens=False):
        return {'entailment</answer>':[1,4], 'neutral</answer>':[2,4],
                'contradiction</answer>':[3,5,4],'<think>':[6]}[text]


@pytest.mark.parametrize('mode',['marginal_accuracy','marginal_f1'])
def test_label_marginal_modes_ignore_sampled_answer_and_replay_direct_gradient(torch,monkeypatch,tmp_path,mode):
    from train.label_marginal import example_rloo
    from train.native_allocation import allocation_surrogate
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    assert trainer.label_marginal_enabled is True
    assert batch['allocation_group_count'].item()==6
    assert batch['allocation_jacobian'].abs().sum()>0
    expected_adv=example_rloo(trainer._label_marginal_rewards,
                              [(r['group_key'],r['slot']) for r in rows],[1]*6)
    assert batch['advantages'].tolist()==pytest.approx(expected_adv)
    cfg['native_pilot']['answer_ce_weight']=0.
    replay=allocation_surrogate(batch['allocation_scores'],batch['allocation_scores'],
                               batch['allocation_jacobian'],batch['allocation_value_share'],
                               batch['allocation_group_count'])
    loss=trainer._compute_loss(trainer.model,batch)
    policy=trainer.model.embed.weight.square().mean()*.01
    assert loss.item()==pytest.approx((policy+.2*replay).item(),abs=1e-6)
    assert torch.autograd.grad(loss-policy,trainer.model.embed.weight)[0].abs().sum()>0
    changed=[{**row,'raw_output':'<think>x</think><answer>entailment</answer>'} for row in rows]
    before=list(trainer._label_marginal_rewards)
    with torch.no_grad():trainer._generate_and_score_completions(changed)
    assert trainer._label_marginal_rewards==pytest.approx(before)


def trainer_fixture(torch,monkeypatch,tmp_path,mode,allocation_loss_weight=.2):
    import trl
    from train.native_grpo import make_native_trainer
    class TinyModel(torch.nn.Module):
        is_gradient_checkpointing=False
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(32,32)
            with torch.no_grad():
                self.embed.weight.copy_(torch.arange(1024).reshape(32,32).sin()/5)
        def get_input_embeddings(self):return self.embed
        def forward(self,input_ids=None,attention_mask=None,use_cache=False,logits_to_keep=None,inputs_embeds=None):
            embedded=self.embed(input_ids) if inputs_embeds is None else inputs_embeds
            logits=(embedded*attention_mask.unsqueeze(-1)).cumsum(1)
            return SimpleNamespace(logits=logits[:,-logits_to_keep:])
    class BaseTrainer:
        def _compute_loss(self,model,inputs):
            return model.embed.weight.square().mean()*.01
        def _generate_and_score_completions(self,inputs):
            self._calculate_rewards(inputs,None,None,None)
            return dict(completion_mask=torch.ones(len(inputs),8),
                        tool_mask=torch.tensor([[0,1,1,0,1,1,0,0]]*len(inputs)),
                        advantages=torch.arange(len(inputs)).float()+1)
    monkeypatch.setattr(trl,'GRPOTrainer',BaseTrainer)
    cfg={'seed':0,'native_mode':mode,'native_pilot':dict(groups_per_update=1,num_generations=2,
        score_microbatch=2,allocation_weight=.2,allocation_loss_weight=allocation_loss_weight,
        answer_ce_weight=.1,disable_dropout=True,reasoning_budget=4,saliency_floor=.25)}
    cls=make_native_trainer(cfg,tmp_path)
    trainer=cls()
    trainer.model=TinyModel();trainer.processing_class=Tokenizer()
    trainer.accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=1)
    trainer.state=SimpleNamespace(global_step=0);trainer.native_group_order=[7]
    trainer.args=SimpleNamespace(gradient_checkpointing_kwargs=None)
    trainer._metrics={'train':defaultdict(list)}
    trainer._checked_gradients=True
    rows=[]
    for slot,gold in enumerate(('entailment','neutral','contradiction')):
        for rollout in range(2):
            rows.append(dict(group_key=7,slot=slot,slot_weight=1,prompt_ids_native=[11+slot],
                prefix_ids=[6,20+rollout,22+rollout,8],gold=gold,raw_output='<think>x</think><answer>neutral</answer>',
                reasoning_tokens=2,id=str(slot),dataset='vianli',premise='P',hypothesis='H',label=gold,topic='01'))
    return trainer,cfg,rows


@pytest.mark.parametrize('mode',['allocation','joint'])
def test_direct_modes_transport_allocation_cache_through_generation(torch,monkeypatch,tmp_path,mode):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    assert batch['allocation_scores'].shape==(6,3)
    assert batch['allocation_jacobian'].shape==(6,3)
    assert batch['allocation_value_share'].shape==(6,)
    assert batch['allocation_group_count'].ndim==0
    assert batch['allocation_group_count'].item()==2
    assert not batch['allocation_jacobian'].requires_grad
    assert torch.isfinite(batch['allocation_jacobian']).all()


@pytest.mark.parametrize('mode',['allocation','joint'])
def test_direct_loss_changes_student_parameter_gradient_and_preserves_cached_value(torch,monkeypatch,tmp_path,mode):
    from train.native_allocation import allocation_surrogate
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    cfg['native_pilot']['answer_ce_weight']=0.
    expected=allocation_surrogate(batch['allocation_scores'],batch['allocation_scores'],
        batch['allocation_jacobian'],batch['allocation_value_share'],batch['allocation_group_count'])
    total=trainer._compute_loss(trainer.model,batch)
    policy=trainer.model.embed.weight.square().mean()*.01
    assert total.item()==pytest.approx((policy+.2*expected).item(),abs=1e-6)
    allocation_grad=torch.autograd.grad(total-policy,trainer.model.embed.weight)[0]
    assert allocation_grad.abs().sum().item()>0


def test_direct_microbatch_shuffle_matches_full_model_gradient(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'allocation')
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    cfg['native_pilot']['answer_ce_weight']=0.
    whole=trainer._compute_loss(trainer.model,batch)-trainer.model.embed.weight.square().mean()*.01
    whole_grad=torch.autograd.grad(whole,trainer.model.embed.weight)[0]
    partial=trainer.model.embed.weight.sum()*0
    for indices in ([4,1],[0,5],[3,2]):
        part={key:value[indices] if torch.is_tensor(value) and value.ndim>0 and value.shape[0]==6 else value
              for key,value in batch.items()}
        partial=partial+trainer._compute_loss(trainer.model,part)-trainer.model.embed.weight.square().mean()*.01
    grad=torch.autograd.grad(partial,trainer.model.embed.weight)[0]
    assert partial.item()==pytest.approx(whole.item(),abs=1e-6)
    assert torch.allclose(grad,whole_grad,atol=1e-6,rtol=1e-5)


def test_reward_channel_and_direct_channel_are_separate(torch,monkeypatch,tmp_path):
    direct,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'allocation')
    with torch.no_grad():a=direct._calculate_rewards(rows,None,None,None)
    joint,cfg2,rows2=trainer_fixture(torch,monkeypatch,tmp_path,'joint')
    with torch.no_grad():b=joint._calculate_rewards(rows2,None,None,None)
    native,cfg3,rows3=trainer_fixture(torch,monkeypatch,tmp_path,'native')
    with torch.no_grad():c=native._calculate_rewards(rows3,None,None,None)
    assert a[:,0].tolist()==pytest.approx([.2,.2,1.2,1.2,.2,.2])
    assert torch.all(b<a)
    assert torch.allclose(b,c)
    assert not getattr(native,'_native_allocation',None)


def test_saliency_cache_has_finite_normalized_credit_and_no_parameter_gradients(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'saliency')
    with torch.no_grad():trainer._calculate_rewards(rows,None,None,None)
    credit=trainer._native_credit
    assert credit.shape==(6,4) and torch.isfinite(credit).all()
    assert torch.allclose(credit[:,:2].sum(-1),torch.full((6,),2.))
    assert torch.equal(credit[:,2:],torch.zeros(6,2))
    assert not credit.requires_grad and trainer.model.embed.weight.grad is None


def test_saliency_mode_transports_token_advantages_and_preserves_answer_credit(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'saliency')
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    assert batch['advantages'].shape==(6,8)
    original=torch.arange(6).float()+1
    assert torch.allclose(batch['advantages'][:,1:3].sum(-1),2*original)
    assert torch.equal(batch['advantages'][:,4],original)
    assert torch.equal(batch['advantages'][:,5],original)
    assert 'allocation_jacobian' in batch
    records=(tmp_path/'rollouts_train.jsonl').read_text().splitlines()
    import json
    assert [json.loads(line)['advantage'] for line in records]==original.tolist()


@pytest.mark.parametrize('mode',['trace_swap','trace_saliency'])
def test_rotated_reward_uses_each_groups_original_minus_rotated_logq(torch,monkeypatch,tmp_path,mode):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    for row in rows:row['prefix_ids'][1]+=row['slot']
    with torch.no_grad():reward=trainer._calculate_rewards(rows,None,None,None)[:,0]
    trace=trainer._native_trace_logq
    base=torch.tensor([.2,.2,1.2,1.2,.2,.2])
    keys=[(7,i%2) for i in range(6)]
    expected=base+.2*torch.tensor([trace['original'][key]-trace['rotated'][key] for key in keys])
    assert torch.allclose(reward,expected,atol=1e-6)
    assert trainer._native_allocation['jacobian'].shape==(6,3)
    assert trainer.model.embed.weight.grad is None


def test_identical_reasoning_prefixes_give_zero_swap_contrast(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'trace_swap')
    with torch.no_grad():reward=trainer._calculate_rewards(rows,None,None,None)[:,0]
    trace=trainer._native_trace_logq
    assert trace['original']==pytest.approx(trace['rotated'],abs=1e-7)
    assert reward.tolist()==pytest.approx([.2,.2,1.2,1.2,.2,.2],abs=1e-7)


def test_swap_reward_keeps_direct_allocation_cache_identical(torch,monkeypatch,tmp_path):
    direct,_,rows=trainer_fixture(torch,monkeypatch,tmp_path,'allocation')
    rotated,_,rows2=trainer_fixture(torch,monkeypatch,tmp_path,'trace_swap')
    for group in (rows,rows2):
        for row in group:row['prefix_ids'][1]+=row['slot']
    with torch.no_grad():
        direct._calculate_rewards(rows,None,None,None)
        rotated._calculate_rewards(rows2,None,None,None)
    for key in ('scores','jacobian','value_share','group_count'):
        assert torch.equal(direct._native_allocation[key],rotated._native_allocation[key])


def test_swap_saliency_preserves_token_advantage_transport(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'trace_saliency')
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    assert batch['advantages'].shape==(6,8)
    assert 'reasoning_credit' in batch and 'allocation_jacobian' in batch
    assert torch.equal(batch['advantages'][:,4],torch.arange(6).float()+1)


def test_replay_diagnostic_excludes_inactive_zeroed_cache_rows(torch,monkeypatch,tmp_path):
    trainer,_,rows=trainer_fixture(torch,monkeypatch,tmp_path,'allocation')
    for row in rows:
        if row['slot']==2:row['slot_weight']=0
    with torch.no_grad():trainer._calculate_rewards(rows,None,None,None)
    batch={'ce_'+key:value for key,value in trainer._native_ce.items()}
    batch.update(ce_weights=trainer._native_weights,
                 ce_num_items=(trainer._native_ce['labels']!=-100).sum())
    batch.update({'allocation_'+key:value for key,value in trainer._native_allocation.items()})
    trainer._compute_loss(trainer.model,batch)
    metrics=trainer._metrics['train']
    assert metrics['native/replay_score_max_difference'][-1]>0
    assert metrics['native/replay_active_score_max_difference'][-1]==pytest.approx(0.,abs=1e-6)


@pytest.mark.parametrize('mode',['allocation','trace_swap','gated_trace_swap'])
def test_shuffled_replay_retains_the_cached_sequence_padding_width(torch,monkeypatch,tmp_path,mode):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    cfg['native_pilot']['score_microbatch']=6
    for i,row in enumerate(rows):
        # Prompt and prefix maxima belong to different examples. The trace
        # cache must retain their conservative width through a shorter slice.
        row['prompt_ids_native']+=[11]*(5-i)
        row['prefix_ids']=[17]*i+row['prefix_ids']
    forward=trainer.model.forward
    def length_sensitive_forward(input_ids=None,**kwargs):
        result=forward(input_ids=input_ids,**kwargs)
        # Model batch-shape sensitivity stands in for BF16 Gated DeltaNet drift.
        result.logits=result.logits+input_ids.shape[1]*torch.arange(32)*.01
        return result
    trainer.model.forward=length_sensitive_forward
    with torch.no_grad():trainer._calculate_rewards(rows,None,None,None)
    selected=torch.tensor([2,0])
    batch={'ce_'+key:value[selected] for key,value in trainer._native_ce.items()}
    batch.update(ce_weights=trainer._native_weights[selected],
                 ce_num_items=(trainer._native_ce['labels']!=-100).sum())
    batch.update({'allocation_'+key:value[selected] if value.ndim else value
                  for key,value in trainer._native_allocation.items()})
    trainer._compute_loss(trainer.model,batch)
    assert trainer._metrics['train']['native/replay_active_score_max_difference'][-1]==pytest.approx(0.,abs=1e-6)


def test_gated_joint_rewards_only_correct_answers_and_retains_direct_loss(torch,monkeypatch,tmp_path):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'gated_joint')
    with torch.no_grad():reward=trainer._calculate_rewards(rows,None,None,None)[:,0]
    assert reward[[0,1,4,5]].tolist()==pytest.approx([.2]*4)
    assert torch.all(reward[[2,3]]>1.2) and torch.all(reward[[2,3]]<=1.4)
    assert trainer._native_allocation['jacobian'].abs().sum()>0
    assert trainer._metrics['train']['native/policy_allocation_bonus'][0]>0


def test_allocation_only_control_removes_policy_loss_without_removing_dense_gradient(torch,monkeypatch,tmp_path):
    from train.native_allocation import allocation_surrogate
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'allocation_only')
    with torch.no_grad():batch=trainer._generate_and_score_completions(rows)
    cfg['native_pilot']['answer_ce_weight']=0.
    expected=allocation_surrogate(batch['allocation_scores'],batch['allocation_scores'],
        batch['allocation_jacobian'],batch['allocation_value_share'],batch['allocation_group_count'])
    total=trainer._compute_loss(trainer.model,batch)
    assert total.item()==pytest.approx((.2*expected).item(),abs=1e-6)
    assert torch.autograd.grad(total,trainer.model.embed.weight)[0].abs().sum()>0


def test_gated_trace_policy_uses_bounded_swap_probability_and_retains_allocation(torch,monkeypatch,tmp_path):
    from train.native_reward import bounded_correct_swap_rewards
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,'gated_trace_swap')
    for row in rows:row['prefix_ids'][1]+=row['slot']
    with torch.no_grad():reward=trainer._calculate_rewards(rows,None,None,None)[:,0]
    trace=trainer._native_trace_logq
    expected=bounded_correct_swap_rewards([.2,.2,1.2,1.2,.2,.2],[7]*6,[i%2 for i in range(6)],
        trace['original'],trace['rotated'],[0,0,1,1,0,0],{(7,0):1,(7,1):1})
    assert reward.tolist()==pytest.approx(expected,abs=1e-6)
    assert reward[[0,1,4,5]].tolist()==pytest.approx([.2]*4)
    assert torch.all(reward[[2,3]]>=1.) and torch.all(reward[[2,3]]<=1.4)
    assert trainer._native_allocation['jacobian'].abs().sum()>0
    assert trainer.model.embed.weight.grad is None


@pytest.mark.parametrize('mode',['trace_swap','trace_saliency','gated_trace_swap'])
def test_original_swapped_and_saliency_forwards_share_a_conservative_global_width(torch,monkeypatch,tmp_path,mode):
    trainer,cfg,rows=trainer_fixture(torch,monkeypatch,tmp_path,mode)
    for i,row in enumerate(rows):
        row['prompt_ids_native']+=[11]*(5-i)
        row['prefix_ids']=[17]*i+row['prefix_ids']
    expected=max(len(row['prompt_ids_native']) for row in rows)+max(len(row['prefix_ids']) for row in rows)+4
    widths=[];forward=trainer.model.forward
    def observed_forward(input_ids=None,inputs_embeds=None,**kwargs):
        widths.append((input_ids if input_ids is not None else inputs_embeds).shape[1])
        return forward(input_ids=input_ids,inputs_embeds=inputs_embeds,**kwargs)
    trainer.model.forward=observed_forward
    with torch.no_grad():trainer._calculate_rewards(rows,None,None,None)
    assert widths and set(widths)=={expected}
