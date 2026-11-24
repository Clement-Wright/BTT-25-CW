import math
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from btt25.rescoring import DistilGPT2Scorer, cheap_features, select_low_confidence


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text.startswith(" ")
        return [{"a": 1, "b": 2}[word] for word in text.split()]


class MarkovLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("table", torch.tensor([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.6, 0.3, 0.1],
        ]).log())
        self.config = SimpleNamespace(n_positions=8, eos_token_id=0)

    def forward(self, input_ids, *, attention_mask, use_cache, return_dict):
        assert use_cache is False
        assert return_dict is True
        assert input_ids.shape == attention_mask.shape
        return SimpleNamespace(logits=self.table[input_ids] + self.anchor)


def scorer(**kwargs):
    return DistilGPT2Scorer(model=MarkovLM(), tokenizer=ToyTokenizer(), **kwargs)


def test_scores_all_text_targets_and_eos_conditioned_on_bos():
    result = scorer().score("a b")
    assert result["log_likelihood"] == pytest.approx(math.log(0.054), abs=1e-6)
    assert result["token_count"] == 3
    assert result["elapsed_seconds"] >= 0


def test_empty_candidate_scores_only_eos_after_bos():
    result = scorer().score("")
    assert result["log_likelihood"] == pytest.approx(math.log(0.5), abs=1e-6)
    assert result["token_count"] == 1


def test_padding_preserves_short_candidate_likelihood_and_eos_target():
    model = scorer()
    alone = model.score("a")
    batched = model.score_many(["a b", "a"], batch_size=2)[1]
    assert batched["log_likelihood"] == pytest.approx(math.log(0.06), abs=1e-6)
    assert batched["log_likelihood"] == pytest.approx(alone["log_likelihood"])
    assert batched["token_count"] == alone["token_count"] == 2


def test_context_limit_rejects_full_candidate_instead_of_truncating():
    model = scorer()
    assert model.score("a a a a a a")["token_count"] == 7
    with pytest.raises(ValueError, match="context"):
        model.score("a a a a a a a")


@pytest.mark.parametrize("text", ["a ", " a"])
def test_candidate_whitespace_is_preserved_after_the_fixed_leading_space(text):
    class ExactTokenizer(ToyTokenizer):
        def encode(self, value, *, add_special_tokens):
            return [2] if value == " " + text else [1]

    model = DistilGPT2Scorer(model=MarkovLM(), tokenizer=ExactTokenizer())
    assert model.score(text)["log_likelihood"] == pytest.approx(math.log(0.12), abs=1e-6)


def test_local_loading_requires_matching_snapshot_hashes_before_model_import(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        DistilGPT2Scorer(tmp_path)
    entries = []
    for name in [
        "model.safetensors", "config.json", "vocab.json", "merges.txt",
        "tokenizer.json", "tokenizer_config.json",
    ]:
        (tmp_path / name).write_bytes(b"fixture")
        entries.append({"file": name, "sha256": hashlib.sha256(b"fixture").hexdigest()})
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_id": "distilbert/distilgpt2",
        "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "files": entries,
    }))
    with pytest.raises(ValueError, match="checksum"):
        DistilGPT2Scorer(tmp_path)


def test_candidate_scoring_keeps_original_text_and_decoder_score():
    original = [{"text": "a b", "score": 12.0, "extra": "keep"}]
    result = scorer().score_candidates(original)
    assert original == [{"text": "a b", "score": 12.0, "extra": "keep"}]
    assert result[0]["text"] == "a b"
    assert result[0]["score"] == 12.0
    assert result[0]["lm_log_likelihood"] == pytest.approx(math.log(0.054), abs=1e-6)
    assert result[0]["lm_token_count"] == 3


def test_cpu_fallback_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        scorer(device="cuda")
    model = scorer(device="cuda", allow_cpu_fallback=True)
    assert model.metadata["device"] == "cpu"
    assert model.metadata["requested_device"] == "cuda"
    assert model.metadata["cpu_fallback_reason"]
    assert model.metadata["dtype"] == "float32"


def test_feature_margin_uses_top_scores_and_best_candidate_word_count():
    features = cheap_features([
        {"text": "a b", "score": 6.0},
        {"text": "a b a b", "score": 10.0},
        {"text": "b", "score": 1.0},
    ], input_length=200)
    assert features["top_two_margin_per_word"] == 1.0
    assert features["top_score_per_word"] == 2.5
    assert features["word_count"] == 4
    assert features["input_length"] == 200


def test_missing_runner_up_has_no_measured_margin_and_score_alias_is_explicit():
    one = cheap_features([{"text": "a", "compact_score": 2.0}], input_length=10)
    assert one["top_two_margin_per_word"] is None
    assert one["top_score_per_word"] == 2.0
    assert cheap_features([], input_length=10)["top_two_margin_per_word"] is None
    with pytest.raises(ValueError, match="score"):
        cheap_features([{"text": "a", "score": float("nan")}], input_length=10)


def test_gate_uses_exact_fraction_low_margin_and_order_independent_hash_ties():
    rows = [
        {"trial_id": "a", "top_two_margin_per_word": None},
        {"trial_id": "b", "top_two_margin_per_word": 0.1},
        {"trial_id": "c", "top_two_margin_per_word": 0.1},
        {"trial_id": "d", "top_two_margin_per_word": 0.8},
    ]
    chosen = select_low_confidence(rows, fraction=0.5, seed=9)
    assert len(chosen) == 2
    assert chosen[0] == "a"
    assert set(chosen[1:]) <= {"b", "c"}
    assert chosen == select_low_confidence(list(reversed(rows)), fraction=0.5, seed=9)
    assert select_low_confidence(rows, fraction=0.0) == []
    assert len(select_low_confidence(rows, fraction=0.49)) == 1
    with pytest.raises(ValueError):
        select_low_confidence(rows, fraction=1.01)
    with pytest.raises(ValueError, match="duplicate"):
        select_low_confidence([rows[0], rows[0]], fraction=0.5)
