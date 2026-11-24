import json
from pathlib import Path
import pytest

from btt25.data import object_hash
from btt25.freeze import current_hashes, seal, validate_score_inputs


def fixture_registry(tmp_path):
    split = {'dev_tune': ['a'], 'dev_report': ['b']}
    split['manifest_hash'] = object_hash(split)
    (tmp_path/'split.json').write_text(json.dumps(split))
    for name in ('config.yaml','checkpoint.pt','predictions.jsonl','a.yaml','a.pt','a.jsonl'):
        (tmp_path/name).write_text(name)
    def group(prefix=''):
        return {'kind':'files','config':{'settings':str(tmp_path/(prefix+'config.yaml' if not prefix else 'a.yaml'))},
                'checkpoint':{'weights':str(tmp_path/('checkpoint.pt' if not prefix else 'a.pt'))},
                'assets':{'predictions':str(tmp_path/('predictions.jsonl' if not prefix else 'a.jsonl'))}}
    return {'schema_version':1,'split_manifest':str(tmp_path/'split.json'),
            'artifacts':{'baseline':group(),'studies':{'session_calibration':group('a'),'confidence_lm':group('a')}}}


def test_live_files_change_hashes_and_sealed_references_reject_mutation(tmp_path):
    registry=fixture_registry(tmp_path)
    before=current_hashes(registry)
    frozen=seal(registry,tmp_path/'freeze.json','Fixed configurations; no reporting selection')
    assert before['baseline']==frozen['baseline']
    validate_score_inputs(registry, frozen, tmp_path/'predictions.jsonl', tmp_path/'config.yaml')
    (tmp_path/'a.pt').write_text('changed weights')
    assert current_hashes(registry)['studies']!=before['studies']
    with pytest.raises(ValueError,match='Frozen hash mismatch'):
        validate_score_inputs(registry,frozen,tmp_path/'predictions.jsonl',tmp_path/'config.yaml')


def test_bare_expected_hashes_and_unbound_predictions_are_rejected(tmp_path):
    registry=fixture_registry(tmp_path)
    expected=current_hashes(registry)
    with pytest.raises(ValueError,match='artifact registry'):
        current_hashes(expected)
    frozen=seal(registry,tmp_path/'freeze.json','fixed')
    (tmp_path/'other.jsonl').write_text('altered output')
    with pytest.raises(ValueError,match='not bound'):
        validate_score_inputs(registry,frozen,tmp_path/'other.jsonl',tmp_path/'config.yaml')


def test_freeze_is_immutable(tmp_path):
    registry=fixture_registry(tmp_path)
    seal(registry,tmp_path/'freeze.json','fixed')
    with pytest.raises(ValueError,match='already exists'):
        seal(registry,tmp_path/'freeze.json','changed')
