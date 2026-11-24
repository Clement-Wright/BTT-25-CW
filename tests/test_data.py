import h5py
import numpy as np
import pytest
from btt25.data import build_manifest, read_trial, freeze_split


def fixture_data(tmp_path):
    root = tmp_path / "t15.fixture"
    root.mkdir()
    for split in ("train", "val", "test"):
        with h5py.File(root / f"data_{split}.hdf5", "w", track_order=True) as file:
            for i in (10, 2, 1):
                g = file.create_group(f"trial_{i}")
                g.create_dataset("input_features", data=np.ones((31+i, 512), np.float32))
                g.attrs["n_time_steps"] = 30
                g.attrs["block_num"] = i
                g.attrs["trial_num"] = i
                g.attrs["session"] = root.name
                if split != "test":
                    g.attrs["seq_len"] = 3
                    g.attrs["sentence_label"] = "A repeated example."
                    g.create_dataset("seq_class_ids", data=[1, 1, 40, 0, 0])
    return tmp_path


def test_lazy_order_lengths_and_labels(tmp_path):
    root = fixture_data(tmp_path)
    manifest = build_manifest(root, ["t15.fixture"])
    val = [r for r in manifest if r["split"] == "val"]
    assert [r["source_group"] for r in val] == ["trial_10", "trial_2", "trial_1"]
    assert all("reference_text" not in r and "phoneme_ids" not in r for r in manifest)
    item = read_trial(val[0])
    assert item["features"].shape == (30, 512)
    assert "phoneme_ids" not in item
    assert read_trial(val[0], include_labels=True)["phoneme_ids"] == [1, 1, 40]
    test = next(r for r in manifest if r["split"] == "test")
    with pytest.raises(ValueError, match="unlabeled"):
        read_trial(test, include_labels=True)


def test_split_grouping_disjoint_and_stable(tmp_path):
    manifest = build_manifest(fixture_data(tmp_path), ["t15.fixture"])
    a = freeze_split(manifest)
    assert a == freeze_split(manifest)
    assert set(a["dev_tune"]).isdisjoint(a["dev_report"])
    assert set(a["dev_tune"]) | set(a["dev_report"]) == {r["trial_id"] for r in manifest if r["split"] == "val"}
    assert a["dev_report"] and a["dev_tune"]


def test_unknown_session_and_bad_features_rejected(tmp_path):
    root = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="session"):
        build_manifest(root, ["other.session"])
    with h5py.File(root / "t15.fixture/data_train.hdf5", "a") as file:
        del file["trial_10/input_features"]
        file["trial_10"].create_dataset("input_features", data=np.ones((30, 8)))
    with pytest.raises(ValueError, match="512"):
        build_manifest(root, ["t15.fixture"])
