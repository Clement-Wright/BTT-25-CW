from copy import deepcopy

import pytest
import torch

from btt25.model import OfficialGRU, gaussian_kernel, smooth_valid
from btt25.v4_rate import RateEncoder, SPECS, geometry


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    source = OfficialGRU(8, 16, 2, 41, n_layers=3).eval()
    x = torch.randn(1, 73, 8)
    return source, x, gaussian_kernel()


def test_baseline_and_two_rate_anchor_parity(fixture):
    base, x, kernel = fixture
    official = base(smooth_valid(x, kernel), [1])
    a = RateEncoder(deepcopy(base), "gru80", kernel).eval()
    pair = RateEncoder(deepcopy(base), "two_rate", kernel).eval()
    torch.testing.assert_close(a(x, 1), official)
    y = pair(x, 1)
    torch.testing.assert_close(y[:, ::2], official)
    torch.testing.assert_close(y[:, 1::2], official[:, :y.shape[1]//2])
    y[:, 1::2].sum().backward()
    assert pair.mid[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", SPECS)
def test_single_trial_lengths_causality_and_delta_restore(fixture, name):
    source, x, kernel = fixture
    enc = RateEncoder(deepcopy(source), name, kernel).eval()
    y = enc(x, 1)
    for length in (47, 48, 68, 73):
        short = enc(x[:, :length], 1)
        assert short.shape == (1, geometry(name, length), 41)
        torch.testing.assert_close(short, y[:, :short.shape[1]], atol=2e-6, rtol=2e-5)
    delta = {k: v.clone() for k, v in enc.trainable_state().items()}
    clone = RateEncoder(deepcopy(source), name, kernel).eval()
    clone.load_delta(delta)
    torch.testing.assert_close(clone(x, 1), y)
    with pytest.raises(ValueError, match="keys"):
        clone.load_delta({})


@pytest.mark.parametrize("name", ["adaptive_filter", "input_transform", "transform_only", "activation_only"])
def test_adapters_begin_at_common_40ms_control(fixture, name):
    source, x, kernel = fixture
    control = RateEncoder(deepcopy(source), "gru40", kernel).eval()
    enc = RateEncoder(deepcopy(source), name, kernel).eval()
    torch.testing.assert_close(enc(x, 1), control(x, 1), atol=2e-6, rtol=2e-5)


def test_bin_grid_contract(fixture):
    source, x, kernel = fixture
    coarse = RateEncoder(deepcopy(source), "bin40", kernel)
    pair = (x[:, :-1:2] + x[:, 1::2]) / 2
    torch.testing.assert_close(coarse.smooth(x), smooth_valid(pair, coarse.kernel))
    held = RateEncoder(deepcopy(source), "bin10_hold", kernel)
    assert held.contract["synthetic_held_grid"]
    torch.testing.assert_close(held.smooth(x), smooth_valid(x.repeat_interleave(2, 1), held.kernel))

def test_cadence_diagnostics_count_both_repeat_neighbors_and_ignore_insertions():
    from btt25.v4_rate_analysis import emission_stats, repeated_phone_counts
    import numpy as np
    labels = [1, 2, 2, 4, 3]
    counts = repeated_phone_counts(labels, [1, 2, 4, 3])
    assert counts["adjacent_repeat_reference"] == 2 and counts["adjacent_repeat_deleted"] == 1
    assert counts["other_reference"] == 3
    assert repeated_phone_counts(labels, [40, *labels])["other_reference"] == 3
    logits = np.zeros((7, 41))
    logits[0, 3] = 4
    _, counts = emission_stats(logits)
    assert counts["blank_frames"] == 6
    assert 0 < counts["sum_entropy_nats"] < 7*np.log(41)



@pytest.mark.parametrize("length", [1, 2, 3, 4, 19, 36])
def test_rate_aware_posterior_pooling(length):
    import numpy as np
    from btt25.ctc import greedy_ctc
    from btt25.v4_rate_decode import transform
    rng = np.random.default_rng(811)
    logits = rng.uniform(-100, 100, (length, 41)).astype(np.float32)
    logits[:min(2, length), 10] += 300
    if length >= 4:
        logits[2, 0] += 400
        logits[3, 10] += 300
    ids = logits.argmax(-1)
    merged = transform(logits, "merge_agree")
    assert greedy_ctc(merged.argmax(-1).tolist()) == greedy_ctc(ids.tolist())
    assert len(merged) == length - (ids[:2*(length//2):2] == ids[1::2]).sum()
    pooled = transform(logits, "pair_average")
    assert len(pooled) == (length + 1)//2
    if length >= 2:
        exp = np.exp(logits.astype(float) - logits.max(-1,keepdims=True))
        p = exp / exp.sum(-1,keepdims=True)
        np.testing.assert_allclose(np.exp(pooled[:length//2]), (p[:2*(length//2):2]+p[1::2])/2,
                                   atol=1e-7,rtol=1e-4)
    np.testing.assert_array_equal(transform(logits, "temperature2").argmax(-1),ids)

def test_initial_frontend_summaries_are_baseline_identity(fixture):
    from btt25.v4_rate_analysis import frontend_parameters
    base, _, kernel = fixture
    model = RateEncoder(deepcopy(base), "adaptive_filter", kernel)
    result = frontend_parameters(model.trainable_state())
    assert result["original_gaussian"]["half_height_width_ms"]["median"] == pytest.approx(95.12, abs=.1)
    for key in ("0", "1", "3"):
        assert result["learned_at_endpoint_difference"][key]["std_ms"]["median"] == pytest.approx(
            result["original_gaussian"]["std_ms"]["median"], abs=1e-5)
    model = RateEncoder(deepcopy(base), "input_transform", kernel)
    result = frontend_parameters(model.trainable_state())
    assert result["residual_spectral_norm"] == 0 and result["residual_relative_frobenius_to_identity"] == 0
    assert result["softsign_plus_coefficient_times_tanh_minus_softsign"]["median"] == 0

def test_rate_decoder_scores_only_allowed_rows_and_keeps_empty_edits():
    from btt25.v4_rate_decode import score
    trial = {"trial_id": "tune", "session_id": "s", "corpus_id": "c"}
    pred = {"text": "", "phones": [], "decoder_frames": 3, "original_frames": 6}
    blind = {"tune": pred, "report": {"text": "unused"}}
    refs = {"tune": {"reference_text": "one two", "phoneme_ids": [3,3]}}
    m, rows = score(blind, [trial], refs, blind, blind, "native")
    assert m["counts"]["word_deletions"] == 2 and m["counts"]["phone_deletions"] == 2
    assert m["counts"]["empty"] == 1 and m["wer"] == m["per"] == 1
    assert [r["trial_id"] for r in rows] == ["tune"]

def test_native_lexical_replay_audit_preserves_phone_identity():
    from btt25.v4_rate_decode import score
    trial={"trial_id":"t","session_id":"s"}
    pred={"text":"a","phones":[1],"decoder_frames":3,"original_frames":4}
    rows={"t":pred}; old={"t":{**pred,"text":"b"}}
    refs={"t":{"reference_text":"a","phoneme_ids":[1]}}
    with pytest.raises(ValueError,match="did not replay"):
        score(rows,[trial],refs,rows,old,"native")
    differences=[]
    metric,_=score(rows,[trial],refs,rows,old,"native",lexical_disagreements=differences)
    assert len(differences)==1 and metric["counts"]["word_errors"]==0
    with pytest.raises(ValueError,match="phones differ"):
        score(rows,[trial],refs,rows,{"t":{**pred,"phones":[2]}},"native",lexical_disagreements=[])

def test_retrained_two_rate_loader_requires_declared_hash_and_matching_contract(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from btt25 import two_rate_expanded as study

    root = tmp_path / "two_rate"
    path = root / "checkpoints" / "last.pt"
    path.parent.mkdir(parents=True)
    torch.save({"condition": "two_rate", "source_sha256": "official", "step": 2048,
                "trainable": {"example": torch.tensor([1.0])}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"status": "complete", "updates": 2048, "checkpoint_sha256": digest,
                "contract": {"condition": "two_rate"}}
    metrics = root / "metrics.json"
    metrics.write_text(json.dumps(metadata))
    config = {"checkpoint": "weights", "model_args": "args", "checkpoint_hash": "official",
              "contract_hash": "parent-contract"}
    parent = SimpleNamespace(contract_hash="parent-contract")
    prep = {"dataset": {"data_transforms": {"smooth_kernel_std": 1, "smooth_kernel_size": 3}}}
    monkeypatch.setattr(study, "load_encoder", lambda *args: (parent, prep))
    module = SimpleNamespace(contract=metadata["contract"], load_delta=lambda delta: None)
    monkeypatch.setattr(study, "RateEncoder", lambda *args: module)
    module.eval = lambda: module
    module.cuda = lambda: module
    with pytest.raises(ValueError, match="Unverified"):
        study.load_trained(config, root)
    assert study.load_trained(config, root, digest) is module
    metadata["contract"] = {"condition": "different"}
    metrics.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="architecture"):
        study.load_trained(config, root, digest)
