import json
import numpy as np
import pytest
from btt25.data import object_hash, load_split_manifest
from btt25.inference import recover_cached_predictions


def test_edited_or_leaking_split_fails_before_reference_access(tmp_path):
    value={"dev_tune":["a"],"dev_report":["b"]}
    value["manifest_hash"]=object_hash(value)
    p=tmp_path/"splits.json"
    p.write_text(json.dumps(value))
    assert load_split_manifest(p)["dev_tune"]==["a"]
    value["dev_tune"].append("b")
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError): load_split_manifest(p)


def test_inference_resume_requires_old_manifest_and_verified_logits(tmp_path):
    path=tmp_path/"logits"
    path.mkdir()
    array=np.zeros((2,41),dtype=np.float32)
    np.save(path/"x.npy",array)
    row={"trial_id":"a","logit_path":"logits/x.npy","output_length":2,"phoneme_ids":[],"logit_sha256":"wrong","status":"ok"}
    (tmp_path/"predictions.jsonl").write_text(json.dumps(row)+"\n")
    expected={"checkpoint_hash":"test"}
    with pytest.raises(ValueError,match="manifest"):
        recover_cached_predictions(tmp_path,expected,{"a"})
    (tmp_path/"cache_manifest.json").write_text(json.dumps(expected))
    with pytest.raises(ValueError,match="hash"):
        recover_cached_predictions(tmp_path,expected,{"a"})
