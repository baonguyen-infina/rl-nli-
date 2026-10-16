"""Single-sequence greedy Transformers evaluation, independent of vLLM state caches."""


def resolve_model_snapshot(model_name, revision):
    if revision is None:
        return model_name
    import re
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('Evaluation requires an immutable model revision')
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_name,revision=revision,local_files_only=True)


def trim_generated_tokens(ids, tok, stops):
    ids=list(ids)
    if tok.eos_token_id in ids:
        ids=ids[:ids.index(tok.eos_token_id)]
    text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    positions=[text.find(marker) for marker in stops if marker in text]
    if positions:
        boundary=min(positions)
        for at in range(len(ids)):
            prefix=tok.decode(ids[:at+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if len(prefix)>boundary:
                return ids[:at]
    return ids


def greedy_outputs(model,tok,messages_list,max_new_tokens,reasoning_budget=None,answer_scores=None):
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from train.common import chat_template_ids
    if type(max_new_tokens) is not int or max_new_tokens<1:
        raise ValueError('Positive completion budget required')
    if answer_scores is not None and reasoning_budget is None:
        raise ValueError('Answer-readout diagnostics require bounded reasoning')
    opener=tok.encode('<think>',add_special_tokens=False)
    bridge=tok.encode('</think><answer>',add_special_tokens=False)
    ending=tok.encode('</answer>',add_special_tokens=False)+[tok.eos_token_id]
    if reasoning_budget is not None and (type(reasoning_budget) is not int or reasoning_budget<1 or
            reasoning_budget+len(opener)+len(bridge)+len(ending)>=max_new_tokens):
        raise ValueError('Reasoning budget leaves no answer tokens')
    device=model.get_input_embeddings().weight.device
    modes=[(m,m.training) for m in model.modules()]

    def generate_one(prefix,budget,stops=(),bad=()):
        class Stop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                text=tok.decode(input_ids[0,len(prefix):].tolist(),skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
                return any(marker in text for marker in stops)
        ids=torch.tensor([prefix],device=device,dtype=torch.long)
        options=dict(max_new_tokens=budget,do_sample=False,num_beams=1,use_cache=True,
                     eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
                     attention_mask=torch.ones_like(ids))
        if stops:options['stopping_criteria']=StoppingCriteriaList([Stop()])
        if bad:options['bad_words_ids']=[tok.encode(text,add_special_tokens=False) for text in bad]
        generated=model.generate(input_ids=ids,**options)[0,len(prefix):].tolist()
        return trim_generated_tokens(generated,tok,stops)
    results=[]
    try:
        model.eval()
        with torch.inference_mode():
            for messages in messages_list:
                prompt=chat_template_ids(tok,messages)
                if reasoning_budget is None:
                    ids=generate_one(prompt,max_new_tokens)
                    results.append((tok.decode(ids,skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False),len(ids)))
                else:
                    reason=generate_one(prompt+opener,reasoning_budget,('</think>','<answer>'),
                                        ('<think>','</answer>'))
                    answer_budget=max_new_tokens-len(opener)-len(reason)-len(bridge)-len(ending)
                    answer=generate_one(prompt+opener+reason+bridge,answer_budget,('</answer>',),
                                        ('<think>','</think>','<answer>'))
                    ids=opener+reason+bridge+answer+ending
                    results.append((tok.decode(ids[:-1],skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),len(ids)))
                    if answer_scores is not None:
                        from eval.readout_probe import canonical_answer_scores
                        audit=canonical_answer_scores(model,tok,prompt+opener+reason+bridge)
                        audit['completion_prefix_length']=len(opener)+len(reason)+len(bridge)
                        answer_scores.append(audit)
    finally:
        for module,training in modes:module.training=training
    return results


def generate_transformers(model_name,messages_list,max_new_tokens,adapter=None,reasoning_budget=None,
                          model_revision=None,answer_scores=None):
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from train.common import load_model_and_tokenizer,patch_torchao_compatibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    patch_torchao_compatibility()
    model,tok=load_model_and_tokenizer(resolve_model_snapshot(model_name,model_revision))
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    model=model.to('cuda').eval()
    return greedy_outputs(model,tok,messages_list,max_new_tokens,reasoning_budget,answer_scores)
