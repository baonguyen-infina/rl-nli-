"""Paired inference uncertainty must resample whole premise groups."""
import numpy as np
import pytest
from analysis.stats import paired_cluster_macro_f1_bootstrap


def score(gold,pred):
    values=[]
    for label in ('entailment','neutral','contradiction'):
        tp=sum(g==p==label for g,p in zip(gold,pred))
        denominator=sum(g==label for g in gold)+sum(p==label for p in pred)
        values.append(2*tp/denominator if denominator else 0.)
    return sum(values)/3


def test_cluster_bootstrap_matches_independent_whole_group_reference():
    gold=['entailment','neutral','contradiction','neutral','entailment']
    a=['entailment','neutral','neutral','contradiction','entailment']
    b=['neutral','neutral','contradiction','neutral',None]
    groups=['P1','P1','P2','P2','P3'];members=[[0,1],[2,3],[4]]
    rng=np.random.default_rng(7);draws=rng.integers(0,3,size=(64,3));diffs=[]
    for draw in draws:
        ix=[i for cluster in draw for i in members[cluster]]
        diffs.append(score([gold[i] for i in ix],[a[i] for i in ix])-score([gold[i] for i in ix],[b[i] for i in ix]))
    out=paired_cluster_macro_f1_bootstrap(gold,a,b,groups,n_boot=64,seed=7)
    assert out['diff']==pytest.approx(score(gold,a)-score(gold,b))
    assert out['ci_low']==pytest.approx(np.percentile(diffs,2.5))
    assert out['ci_high']==pytest.approx(np.percentile(diffs,97.5))
    assert out['n_clusters']==3 and out['bootstrap_unit']=='premise_cluster'


def test_single_premise_has_no_fake_independent_example_variance():
    gold=['entailment','neutral'];a=['entailment','neutral'];b=[None,None]
    out=paired_cluster_macro_f1_bootstrap(gold,a,b,['P','P'],n_boot=20)
    assert out['ci_low']==pytest.approx(out['diff'])
    assert out['ci_high']==pytest.approx(out['diff'])


def test_identical_predictions_have_exact_zero_difference_and_interval():
    out=paired_cluster_macro_f1_bootstrap(['neutral']*3,['neutral']*3,['neutral']*3,['A','A','B'],n_boot=20)
    assert out['diff']==out['ci_low']==out['ci_high']==0.


@pytest.mark.parametrize('gold,a,b,groups,n_boot',[
    ([],[],[],[],10),(['neutral'],[],['neutral'],['P'],10),
    (['neutral'],['neutral'],['neutral'],[],10),
    (['bad'],['neutral'],['neutral'],['P'],10),
    (['neutral'],['neutral'],['neutral'],['P'],0),
    (['neutral'],['neutral'],['neutral'],['P'],True),
])
def test_invalid_cluster_inputs_fail(gold,a,b,groups,n_boot):
    with pytest.raises(ValueError):paired_cluster_macro_f1_bootstrap(gold,a,b,groups,n_boot=n_boot)
