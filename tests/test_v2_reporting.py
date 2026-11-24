import json

import pytest

from btt25.data import object_hash
from btt25.fixed_candidates_v2 import set_weights
from btt25.report_v2 import verify_run


def test_v2_candidate_cache_checks_data_split_and_admission(tmp_path):
    cache = {"data_manifest_sha256": "data", "source_cache_manifest": {"split_hash": "split"},
             "references_loaded": False}
    cache["manifest_hash"] = object_hash(cache)
    row = {"trial_id": "trial", "decoder_cache_hash": cache["manifest_hash"]}
    readings = {"outcome": "complete", "peak_accounted_host_bytes": 10, "peak_cgroup_bytes": 20,
                "host_ceiling_bytes": 30}
    for name, content in (("cache_manifest.json", cache), ("predictions.jsonl", row),
                          ("resources.json", readings), ("run_manifest.json", {"state": "complete", "saved_trials": 1})):
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    assert verify_run(tmp_path, None, [row], "data", "split") == [row]
    with pytest.raises(ValueError, match="different data or split"):
        verify_run(tmp_path, None, [row], "data", "other")
    (tmp_path / "resources.json").write_text(json.dumps({**readings, "peak_cgroup_bytes": 31}), encoding="utf-8")
    with pytest.raises(ValueError, match="do not qualify"):
        verify_run(tmp_path, None, [row], "data", "split")


def test_fixed_likelihood_does_not_add_acquisition_prior():
    candidates = {"t1": [
        {"text": "cut", "component_scores": {"acoustic": -1, "kenlm_contribution": 999}, "neural_logprob": -5},
        {"text": "cat", "component_scores": {"acoustic": -3, "kenlm_contribution": -999}, "neural_logprob": -1}],
        "t2": []}
    chosen, edits = set_weights(candidates, {"t1": ["cat"], "t2": ["always", "retained"]}, [.25, .75, 1.25])
    assert chosen == .75
    assert edits == {.25: 3, .75: 2, 1.25: 2}
