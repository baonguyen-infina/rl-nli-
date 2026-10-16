"""Định nghĩa ma trận E0–E13, A1–A6, B1–B3 (recipe, format, guess, init_from, cờ ablation, priority, seeds) và sinh configs/runs/<run_id>.yaml.
"""
import yaml

from common.io import RUN_CONFIGS, deep_merge

BASE_MODEL = "Qwen/Qwen3.5-2B"
ENCODER_MODEL = "uitnlp/CafeBERT"

RUNS = {
    "D3": dict(recipe="sft", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="DEV", seeds=[0], overrides={}),
    "D16": dict(recipe="gta", format="align", guess=True, model=BASE_MODEL, init_from="D3",
                priority="DEV", seeds=[0], cos_constraint=True, flat_fallback=True,
                schema_examples="balanced", overrides={"grpo": {"max_steps": 200}}),
    "D18": dict(recipe="gta", format="align", guess=True, model=BASE_MODEL, init_from="D3",
                priority="DEV", seeds=[0], cos_constraint=True, flat_fallback=True,
                schema_examples="balanced", rule_supervision=True,
                overrides={"grpo": {"max_steps": 200}}),
    "G0S": dict(recipe="gta", format="align", guess=True, model=BASE_MODEL, init_from="D3",
                priority="P0", seeds=[0], cos_constraint=True, flat_fallback=True,
                schema_examples="balanced", rule_supervision=True,
                overrides={"grpo": {"max_steps": 1}}),
    "G0G": dict(recipe="gta", format="align", guess=True, model=BASE_MODEL, init_from="D3",
                priority="P0", seeds=[0], cos_constraint=True, flat_fallback=True,
                overrides={"grpo": {"max_steps": 1}}),
    "G0R": dict(recipe="grpo", format="label", guess=False, model=BASE_MODEL, init_from=None,
                priority="P0", seeds=[0], overrides={"grpo": {"max_steps": 1}}),
    "G0": dict(recipe="grpo", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P0", seeds=[0], overrides={"grpo": {"max_steps": 1}}),
    "E0": dict(recipe="zs", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E1": dict(recipe="zs", format="cot", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E2": dict(recipe="zs", format="align", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E3": dict(recipe="sft", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E4": dict(recipe="sft", format="cot", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E5": dict(recipe="sft", format="align", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E6": dict(recipe="grpo", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E7": dict(recipe="grpo", format="cot", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E8": dict(recipe="grpo", format="align", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E9": dict(recipe="grpo", format="label", guess=False, model=BASE_MODEL, init_from="E3",
               priority="P2", seeds=[0], overrides={}),
    "E10": dict(recipe="grpo", format="cot", guess=False, model=BASE_MODEL, init_from="E4",
               priority="P1", seeds=[0], overrides={}),
    "E11": dict(recipe="grpo", format="align", guess=False, model=BASE_MODEL, init_from="E5",
               priority="P1", seeds=[0], overrides={}),
    "E12": dict(recipe="grpo", format="cot", guess=True, model=BASE_MODEL, init_from="E4",
               priority="P1", seeds=[0], overrides={}),
    "E13": dict(recipe="grpo", format="align", guess=True, model=BASE_MODEL, init_from="E5",
               priority="P1", seeds=[0], overrides={}),
    "B1": dict(recipe="encoder", format="label", guess=False, model=ENCODER_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
}


# Teacher-free development recipes, separate from the original locked grid.
for dev_id, fmt, guess, flat in (("D9", "label", False, False),
                               ("D10", "cot", False, False),
                               ("D11", "align", False, False),
                               ("D12", "cot", True, False),
                               ("D13", "align", True, False),
                               ("D14", "align", True, True)):
    RUNS[dev_id] = dict(recipe="gta" if guess else "grpo", format=fmt, guess=guess,
                       model=BASE_MODEL, init_from="D3", priority="DEV", seeds=[0],
                       cos_constraint=True, flat_fallback=flat,
                       overrides={"grpo": {"max_steps": 200}})



RUNS['E2B'] = dict(recipe='zs', format='align', guess=True, model=BASE_MODEL,
                    init_from=None, priority='PAPER', seeds=[0], schema_examples='balanced', overrides={})
RUNS['D3A'] = dict(recipe='zs', format='align', guess=True, model=BASE_MODEL,
                    init_from='D3', priority='PAPER', seeds=[0,1,2], schema_examples='balanced', overrides={})

# Matched paper comparisons: the same gold-only warm start and shared schedule.
RUNS['D19'] = dict(recipe='grpo', format='align', guess=False, model=BASE_MODEL,
                    init_from='D3', priority='PAPER', seeds=[0, 1, 2],
                    schema_examples='balanced', overrides={'grpo': {'max_steps': 200}})
for rid, changes in {
    'D20': {'rule_supervision': False, 'flat_fallback': False},
    'D21': {'cos_constraint': False},
    'D23': {'overrides': {'grpo': {'max_steps': 200, 'reward_weights': [1.0, 0.2, 0.0]}}},
}.items():
    RUNS[rid] = {**RUNS['D18'], 'priority': 'PAPER', 'seeds': [0, 1, 2], **changes}



# Registered after observing quote/status binding failures in D18 training rollouts.
RUNS['D24'] = {**RUNS['D18'], 'rule_support': 'schema', 'seeds': [0,1,2]}
RUNS['G0T'] = {**RUNS['D24'], 'priority': 'P0', 'seeds': [0],
              'overrides': {'grpo': {'max_steps': 1}}}
RUNS['D25'] = {**RUNS['D24'], 'cos_constraint': False}
RUNS['D26'] = {**RUNS['D24'], 'overrides': {'grpo': {'max_steps': 200, 'reward_weights': [1.0,0.2,0.0]}}}

# Prespecified teacher-free native-contrast pilot and matched controls.
RUNS['D3R'] = {**RUNS['D3']}
for rid, mode in [('N0', 'ce'), ('N1', 'ordinary'), ('N2', 'native'), ('N3', 'random')]:
    RUNS[rid] = dict(recipe='native', format='cot', guess=False, model=BASE_MODEL,
                    init_from='D3R', priority='DEV', seeds=[0], native_mode=mode,
                    bounded_reasoning=True,
                    overrides={'eval': {'gpu_memory_utilization': 0.65, 'max_new_tokens': {'cot': 96}}})
RUNS['G0N'] = {**RUNS['N2'], 'priority': 'P0',
              'overrides': {'native_pilot': {'max_steps': 1, 'save_steps': 1},
                            'eval': {'gpu_memory_utilization': 0.65, 'max_new_tokens': {'cot': 96}}}}
RUNS['G0N2'] = {**RUNS['G0N']}
RUNS['G0N3'] = {**RUNS['G0N']}
RUNS['N2R'] = {**RUNS['N2']}
RUNS['D3R3'] = {**RUNS['D3']}
RUNS['N2R3'] = {**RUNS['N2'], 'init_from': 'D3R3'}

# Fresh IDs for deterministic scoring; never overwrite the R3 frozen pilot.
for parent in ['N0','N1','N2','N3']:
    RUNS[parent+'R4'] = {**RUNS[parent], 'init_from':'D3R3',
                         'scoring_protocol':'deterministic'}
RUNS['G0N4'] = {**RUNS['G0N3'], 'init_from':'D3R3',
               'scoring_protocol':'deterministic'}

# Allocation channel comparison: keep data, initialization and budgets fixed.
RUNS['N4R4'] = {**RUNS['N2R4'], 'native_mode':'allocation'}
RUNS['N5R4'] = {**RUNS['N2R4'], 'native_mode':'joint'}
RUNS['G0N5'] = {**RUNS['G0N4'], 'native_mode':'allocation'}
RUNS['G0N6'] = {**RUNS['G0N4'], 'native_mode':'joint'}

# Read-only fixed-adapter validation, using one shared runtime preset in base.yaml.
for number in (2,4,5):
    for repeat in ('A','B'):
        RUNS[f'V{number}R4{repeat}'] = {**RUNS[f'N{number}R4'], 'recipe':'zs',
            'init_from':f'N{number}R4', 'validation_protocol':'fixed_single_sequence'}
for repeat in ('A','B'):
    RUNS[f'VDR4{repeat}'] = {**RUNS['N2R4'], 'recipe':'zs', 'init_from':'D3R3',
                           'validation_protocol':'fixed_single_sequence'}
RUNS['VDR4L'] = {**RUNS['VDR4A'], 'format':'label', 'bounded_reasoning':False}
for number in (2,4,5):
    RUNS[f'V{number}R4L'] = {**RUNS[f'V{number}R4A'], 'format':'label', 'bounded_reasoning':False}

# Native token-credit hypothesis and matched200-step controls.
RUNS['N1R5'] = {**RUNS['N1R4']}
RUNS['N4R5'] = {**RUNS['N4R4']}
RUNS['N7R5'] = {**RUNS['N4R4'], 'native_mode':'saliency'}
RUNS['G0N7'] = {**RUNS['G0N5'], 'native_mode':'saliency'}
for number in (1,4,7):
    for repeat in ('A','B'):
        RUNS[f'V{number}R5{repeat}'] = {**RUNS[f'N{number}R5'], 'recipe':'zs',
            'init_from':f'N{number}R5', 'validation_protocol':'fixed_single_sequence'}
    RUNS[f'V{number}R5L'] = {**RUNS[f'V{number}R5A'], 'format':'label','bounded_reasoning':False}

# Context specificity of student reasoning; same shared native pilot budget.
RUNS['N5R6'] = {**RUNS['N5R4']}
RUNS['N8R6'] = {**RUNS['N5R4'], 'native_mode':'trace_swap'}
RUNS['N9R6'] = {**RUNS['N8R6'], 'native_mode':'trace_saliency'}
RUNS['G0N8'] = {**RUNS['G0N6'], 'native_mode':'trace_swap'}
RUNS['G0N9'] = {**RUNS['G0N6'], 'native_mode':'trace_saliency'}
for number in (5,8,9):
    for repeat in ('A','B'):
        RUNS[f'V{number}R6{repeat}'] = {**RUNS[f'N{number}R6'], 'recipe':'zs',
            'init_from':f'N{number}R6', 'validation_protocol':'fixed_single_sequence'}
    RUNS[f'V{number}R6L'] = {**RUNS[f'V{number}R6A'], 'format':'label','bounded_reasoning':False}


# Shared near-full-data schedule; change only the horizon for matched controls.
for number, parent in ((1, 'N1R5'), (4, 'N4R5'), (5, 'N5R6')):
    RUNS[f'N{number}R7'] = {**RUNS[parent], 'native_schedule': 'full_pass'}
    for repeat in ('A', 'B'):
        RUNS[f'V{number}R7{repeat}'] = {**RUNS[f'N{number}R7'], 'recipe': 'zs',
            'init_from': f'N{number}R7', 'validation_protocol': 'fixed_single_sequence'}
    RUNS[f'V{number}R7L'] = {**RUNS[f'V{number}R7A'], 'format': 'label', 'bounded_reasoning': False}


# A5-style capacity check of the existing method; no teacher-derived targets.
RUNS['D4R8'] = {**RUNS['D3R3'], 'model': 'Qwen/Qwen3.5-4B'}
RUNS['N4R8'] = {**RUNS['N4R7'], 'model': 'Qwen/Qwen3.5-4B', 'init_from': 'D4R8'}
RUNS['G0N4R8'] = {**RUNS['G0N5'], 'model': 'Qwen/Qwen3.5-4B', 'init_from': 'D4R8'}
for repeat in ('A', 'B'):
    RUNS[f'V4R8{repeat}'] = {**RUNS['N4R8'], 'recipe': 'zs', 'init_from': 'N4R8',
                           'validation_protocol': 'fixed_single_sequence'}
RUNS['V4R8L'] = {**RUNS['V4R8A'], 'format': 'label', 'bounded_reasoning': False}


# R9 is restricted to the post-trained2B backbone by the user's capacity cap.
# R8 capacity-check entries above are historical, unlaunched configurations.
RUNS['D3R9'] = {**RUNS['D3R3'], 'sft_schedule': 'full_budget'}
for rid, parent, warm in (
    ('N4R9', 'N4R5', 'D3R9'),
    ('N4W9', 'N4R5', 'D3R3'),
    ('N1R9', 'N1R5', 'D3R9'),
):
    RUNS[rid] = {**RUNS[parent], 'model': BASE_MODEL, 'init_from': warm,
                 'scoring_protocol': 'deterministic_fixed_padding',
                 'native_schedule': 'full_reasoning', 'eval_schedule': 'full_reasoning'}
    smoke = 'G0' + rid
    RUNS[smoke] = {**RUNS[rid], 'priority': 'P0', 'native_schedule': 'long_smoke'}
    RUNS[smoke + 'P'] = {**RUNS[smoke]}
    for repeat in ('A', 'B'):
        RUNS['V' + rid[1:] + repeat] = {
            **RUNS[rid], 'recipe': 'zs', 'init_from': rid,
            'validation_protocol': 'fixed_single_sequence'}
    RUNS['V' + rid[1:] + 'L'] = {
        **RUNS['V' + rid[1:] + 'A'], 'format': 'label', 'bounded_reasoning': False}
RUNS['VDR9L'] = {**RUNS['D3R9'], 'recipe': 'zs', 'init_from': 'D3R9',
                 'bounded_reasoning': False, 'validation_protocol': 'fixed_single_sequence',
                 'overrides': {'eval': {'gpu_memory_utilization': 0.65}}}

# Test both channels of the same native critic after the ordinary-RL lane frees.
# Group score reward reaches sampled reasoning; direct replay reaches label scores.
RUNS['N5R10'] = {**RUNS['N4R9'], 'native_mode': 'joint'}
RUNS['G0N5R10P'] = {**RUNS['N5R10'], 'priority': 'P0', 'native_schedule': 'long_smoke'}
for repeat in ('A', 'B'):
    RUNS['V5R10' + repeat] = {**RUNS['N5R10'], 'recipe': 'zs', 'init_from': 'N5R10',
                             'validation_protocol': 'fixed_single_sequence'}
RUNS['V5R10L'] = {**RUNS['V5R10A'], 'format': 'label', 'bounded_reasoning': False}

# Fixed next method and its no-policy-gradient control; no per-cell tuning.
RUNS['N6R11'] = {**RUNS['N5R10'], 'native_mode':'gated_joint'}
RUNS['N0R11'] = {**RUNS['N4R9'], 'native_mode':'allocation_only'}
RUNS['N7R12'] = {**RUNS['N6R11'], 'native_grouping':'random','native_schedule':'prefix_200'}
for rid in ('N6R11','N0R11'):
    RUNS['G0'+rid+'P'] = {**RUNS[rid], 'priority':'P0','native_schedule':'long_smoke'}
    RUNS['V'+rid[1:]+'L'] = {**RUNS[rid], 'recipe':'zs','init_from':rid,
                            'format':'label','bounded_reasoning':False,
                            'validation_protocol':'fixed_single_sequence'}
RUNS['G0N7R12P'] = {**RUNS['N7R12'],'priority':'P0','native_schedule':'long_smoke'}
RUNS['V7R12L'] = {**RUNS['N7R12'],'recipe':'zs','init_from':'N7R12',
                  'format':'label','bounded_reasoning':False,'validation_protocol':'fixed_single_sequence'}

# One shared HF greedy protocol, registered before any held-out evaluation.
# H views first run on dev; the exact same frozen config is then hash-locked.
for source in ('D3R9','N1R9','N4R9','N5R10','N6R11','N0R11','N7R12'):
    for suffix,fmt in (('L','label'),('C','cot')):
        RUNS['H'+source+suffix] = {
            **RUNS[source], 'recipe':'zs','init_from':source,'format':fmt,
            'bounded_reasoning':fmt=='cot','native_schedule':'full_reasoning',
            'eval_schedule':'full_reasoning','validation_protocol':'fixed_single_sequence',
            'overrides':{'eval':{'backend':'transformers','gpu_memory_utilization':0.65,
                'model_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc'}}}
RUNS['HN6R11P200L'] = {**RUNS['HN6R11L'],'init_checkpoint':200}
RUNS['HN7R12P200L'] = {**RUNS['HN7R12L'],'init_checkpoint':200}

# R13 tests finite-label marginalization, without a native allocation objective.
for rid,mode in (('N10R13','marginal_f1'),('N11R13','marginal_accuracy')):
    RUNS[rid]={**RUNS['N4R9'],'native_mode':mode,'native_schedule':'prefix_200',
        'overrides':{'eval':{'backend':'transformers',
            'model_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc'}}}
    RUNS['G0'+rid+'P']={**RUNS[rid],'priority':'P0','native_schedule':'long_smoke'}
    RUNS['H'+rid+'L']={**RUNS[rid],'recipe':'zs','init_from':rid,
        'format':'label','bounded_reasoning':False}


RUNS['P0R14'] = {**RUNS['HD3R9C'], 'priority': 'P0', 'readout_diagnostic': True}

# Fresh full400 schedule; native trainer explicitly generates CoT, while the
# prospectively selected inference interface is the existing greedy label view.
RUNS['N12R15'] = {**RUNS['N7R12'], 'format': 'label', 'training_format': 'cot',
    'bounded_reasoning': False, 'native_schedule': 'full_reasoning',
    'validation_protocol': 'fixed_single_sequence',
    'overrides': {'eval': {'backend': 'transformers', 'gpu_memory_utilization': 0.65,
        'model_revision': '15852e8c16360a2fea060d615a32b45270f8a8fc'}}}
RUNS['G0N12R15P'] = {**RUNS['N12R15'], 'priority': 'P0', 'native_schedule': 'long_smoke'}

# Counterfactual trace-swap credit assignment, matched to N12R15 on every
# backbone, data, rollout, optimizer, training-format and readout setting.
RUNS['N13R16'] = {**RUNS['N12R15'], 'native_mode': 'trace_swap'}
RUNS['G0N13R16P'] = {**RUNS['N13R16'], 'priority': 'P0', 'native_schedule': 'long_smoke'}

# Short trace-saliency follow-up, matched to R15 except attribution and horizon.
RUNS['N14R17'] = {**RUNS['N12R15'], 'native_mode': 'trace_saliency', 'native_schedule': 'prefix_200'}
RUNS['G0N14R17P'] = {**RUNS['N14R17'], 'priority': 'P0', 'native_schedule': 'long_smoke'}

# Bounded correctness-gated probability contrast; later source revision also
# fixes original/rotated/saliency score geometry. Earlier frozen VMs are unchanged.
RUNS['N15R18'] = {**RUNS['N12R15'], 'native_mode': 'gated_trace_swap'}
RUNS['G0N15R18P'] = {**RUNS['N15R18'], 'priority': 'P0', 'native_schedule': 'long_smoke'}

# The same bounded method on shared-premise groups; scheduled examples and
# label-slot templates stay matched to its random-group counterpart.
RUNS['N16R19'] = {**RUNS['N15R18'], 'native_grouping':'native'}
RUNS['G0N16R19P'] = {**RUNS['N16R19'], 'priority':'P0', 'native_schedule':'long_smoke'}

# Registered teacher-free follow-up: full-horizon native token credit and
# macro-F1-aligned label marginalization, both from the same SFT warm start.
RUNS['N20R20'] = {**RUNS['N12R15'], 'native_mode':'trace_saliency',
                  'native_schedule':'full_reasoning'}
RUNS['N21R20'] = {**RUNS['N12R15'], 'native_mode':'marginal_f1',
                  'native_schedule':'full_reasoning'}
RUNS['G0N20R20P'] = {**RUNS['N20R20'], 'priority':'P0', 'native_schedule':'long_smoke'}
RUNS['G0N21R20P'] = {**RUNS['N21R20'], 'priority':'P0', 'native_schedule':'long_smoke'}

# Explicit development readouts of preserved checkpoints after the requested VM
# shutdown. These views do not claim completion of either interrupted parent.
RUNS['HN12R15P200L'] = {**RUNS['N12R15'], 'recipe':'zs', 'init_from':'N12R15', 'init_checkpoint':200}
RUNS['HN13R16P100L'] = {**RUNS['N13R16'], 'recipe':'zs', 'init_from':'N13R16', 'init_checkpoint':100}

# Early readout of the running R13 methods; primary CoT selection stays frozen.
for source in ('N10R13', 'N11R13'):
    RUNS['H'+source+'P100L'] = {**RUNS['H'+source+'L'], 'init_checkpoint': 100}


def build_run_config(run_id: str) -> dict:
    spec = dict(RUNS[run_id])
    overrides = spec.pop("overrides")
    spec["model"] = {"name": spec["model"]}
    spec["run_id"] = run_id
    return deep_merge(spec, overrides)


def write_run_configs() -> list:
    RUN_CONFIGS.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_id in RUNS:
        p = RUN_CONFIGS / f"{run_id}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(build_run_config(run_id), f, allow_unicode=True, sort_keys=False)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_run_configs():
        print(p)
