"""Main paper results separate matched methods, formats and fixed-budget controls."""
import pytest


def row(run,fmt='label',split='test'):
    return dict(run_id=run,seed=0,split=split,format=fmt,n=1000,
                accuracy=.59,macro_f1=.58,parse_rate=1.)


def test_main_table_contains_only_registered_primary_label_views():
    from analysis.paper_report import render_comparison
    values=[row('HD3R9L'),row('HN1R9L'),row('HN6R11L'),
            row('HN6R11P200L'),row('HN6R11C','cot')]
    table=render_comparison(values,'main')
    assert 'SFT' in table and 'bounded' in table
    assert '58.00' in table and '200' not in table and 'reasoning' not in table
    assert 'split=test' in table


def test_ablation_table_reports_fixed_checkpoint_not_best_selection():
    from analysis.paper_report import render_comparison
    table=render_comparison([row('HN6R11L'),row('HN6R11P200L'),row('HN7R12P200L')],'mechanism')
    assert 'Native groups' in table and 'Random groups' in table
    assert 'bounded' not in table


def test_table_rejects_unregistered_or_misformatted_test_view():
    from analysis.paper_report import render_comparison
    for value in (row('unknown'),row('HN6R11L','cot'),row('HN6R11L',split='dev')):
        with pytest.raises(ValueError):render_comparison([value],'main')


def test_secondary_table_is_explicit_and_does_not_mix_primary_scores():
    from analysis.paper_report import render_comparison
    table=render_comparison([row('HD3R9C','cot'),row('HN6R11C','cot'),row('HN6R11L')],'secondary')
    assert 'SFT' in table and 'bounded' in table
    assert table.count('58.00')==2


def test_report_aligns_predictions_by_id_and_resamples_premise_clusters():
    from analysis.paper_report import compare_predictions
    a=[dict(id='uit_Adver_P1_1_11_01',gold='entailment',pred='entailment'),
       dict(id='uit_Adver_P1_2_21_01',gold='neutral',pred='neutral'),
       dict(id='uit_Adver_P2_1_31_01',gold='contradiction',pred='neutral')]
    b=[{**r,'pred':'neutral'} for r in reversed(a)]
    out=compare_predictions(a,b,n_boot=32)
    assert out['n_clusters']==2 and out['bootstrap_unit']=='premise_cluster'
    assert out['n']==3
    assert out['diff']>0
    with pytest.raises(ValueError):compare_predictions(a,b[:-1],n_boot=32)
    with pytest.raises(ValueError):compare_predictions(a,[{**b[0],'gold':'entailment'},*b[1:]],n_boot=32)


def test_report_interpretation_comes_from_test_predictions(tmp_path):
    import json
    from analysis.paper_report import build_report
    from eval.metrics import compute_metrics
    gold=['entailment','neutral','contradiction']
    runs={'HD3R9L':True,'HN1R9L':True,'HN0R11L':True,
          'HN4R9L':False,'HN5R10L':False,'HN6R11L':False,
          'HN6R11P200L':False,'HN7R12P200L':True,
          'HN0R11C':True,'HN6R11C':False}
    rows=[]
    for rid,correct in runs.items():
        out=tmp_path/rid;out.mkdir()
        records=[dict(id=f'uit_Adver_P{i}_1_11_01',gold=label,
                      pred=label if correct else 'neutral',parse_ok=True,topic='01')
                 for i,label in enumerate(gold)]
        (out/'predictions_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        rows.append({**row(rid,fmt='cot' if rid.endswith('C') else 'label'),
                     **compute_metrics(records),'source':str(out),'initial_macro_f1':.99999})
    build_report(rows,tmp_path/'paper',n_boot=20)
    tables=tmp_path/'paper/tables'
    text=(tables/'interpretation_test.tex').read_text()
    assert 'native' in text.lower() and 'policy' in text.lower()
    assert 'do not establish' in (tables/'conclusion_test.tex').read_text()
    assert '.99999' not in text
