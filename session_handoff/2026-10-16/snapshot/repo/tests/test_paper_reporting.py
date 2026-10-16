"""Paper artifacts must never report development or training measurements."""
import sys

import pytest

from analysis import make_figs, make_tables


def _row(split='test'):
    return {'run_id': 'N4R9', 'seed': 0, 'split': split,
            'macro_f1': .58, 'accuracy': .59, 'parse_rate': 1.}


def test_paper_table_rejects_development_results():
    with pytest.raises(ValueError, match='test'):
        make_tables.render([_row('dev')], 'dev')


@pytest.mark.parametrize('split', ['dev', None])
def test_paper_table_requires_test_provenance_on_every_row(split):
    row = _row(split)
    if split is None:
        row.pop('split')
    with pytest.raises(ValueError, match='test'):
        make_tables.render([_row(), row], 'test')


def test_paper_table_rejects_missing_results():
    with pytest.raises(ValueError, match='No completed test'):
        make_tables.render([], 'test')


def test_paper_table_reports_recorded_test_results():
    table = make_tables.render([_row()], 'test')
    assert 'N4R9 & 1 & 58.00 & 59.00 & 100.00' in table
    assert 'split=test' in table


def test_paper_cli_defaults_to_test_without_loading_development(tmp_path, monkeypatch):
    splits = []
    monkeypatch.setattr(sys, 'argv', ['make_tables'])
    monkeypatch.setattr(make_tables, 'ROOT', tmp_path)

    def collect(*, split):
        splits.append(split)
        return [_row()]

    monkeypatch.setattr(make_tables, 'collect_runs', collect)
    make_tables.main()
    assert splits == ['test']
    assert (tmp_path / 'paper/tables/results_test.tex').exists()
    assert not (tmp_path / 'paper/tables/results_dev.tex').exists()


def test_paper_cli_rejects_dev_before_loading_results(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['make_tables', '--split', 'dev'])
    monkeypatch.setattr(make_tables, 'collect_runs', lambda **kw: pytest.fail('must reject dev first'))
    with pytest.raises(SystemExit):
        make_tables.main()


def test_training_diagnostic_output_stays_outside_paper():
    assert make_figs.OUTPUT == make_figs.ROOT / 'artifacts/analysis/figures/training.pdf'
