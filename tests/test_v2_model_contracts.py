import pytest
import torch
from torch import nn

from btt25.alternative_encoders import (
    CausalT15Transformer, DiphoneHead, diphone_id, diphone_objective, diphone_targets,
)
from btt25.qwen_scorer import batched_scores


def test_joint_warm_start_and_marginal_ctc():
    torch.manual_seed(0)
    head = nn.Linear(7, 41)
    joint = DiphoneHead.warm_start(head)
    x = torch.randn(18, 7)
    assert torch.allclose(joint.emissions(x), head(x).log_softmax(-1), atol=2e-6)
    assert diphone_targets([40, 1, 1, 40]) == [diphone_id(40, 40), diphone_id(40, 1),
        diphone_id(1, 1), diphone_id(1, 1), diphone_id(1, 40), diphone_id(40, 40)]
    joint_logits = joint(x)
    mix, context, phone = diphone_objective(joint_logits, [40, 1, 1, 40], .3)
    for objective in (context, phone):
        grad, = torch.autograd.grad(objective, joint.weight, retain_graph=True)
        assert grad.abs().sum() > 0 and torch.isfinite(grad).all()
    mix.backward()
    assert joint.bias.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="infeasible"):
        diphone_objective(joint_logits[:6], [40, 1, 1, 40], .3)


def test_causal_transformer_preprocessing_prefix_and_padding_agree():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = CausalT15Transformer(2, width=24, layers=2, heads=3, window=4, dropout=0).eval()
    features = torch.randn(2, 48, 512)
    batch, lengths = model(features, [0, 1], [31, 48])
    assert lengths == (7, 12)
    short, _ = model(features[:1, :31], [0], [31])
    assert torch.allclose(batch[0, :7], short[0], atol=1e-6)
    changed = features.clone()
    changed[:, 24:] = -100 * changed[:, 24:]
    altered, _ = model(changed, [0, 1], [31, 48])
    assert torch.allclose(batch[:, :6], altered[:, :6], atol=1e-6)


def test_exact_candidate_scoring_masks_padding_and_includes_requested_eos():
    from types import SimpleNamespace

    def tokens(texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": [[int(char) for char in text] for text in texts]}

    class Fixture(nn.Module):
        def forward(self, input_ids, attention_mask, use_cache):
            assert use_cache is False
            logits = torch.nn.functional.one_hot((input_ids + 1) % 4, num_classes=4).float()
            return SimpleNamespace(logits=logits)

    model = Fixture()
    finished = batched_scores(model, tokens, ["12", "2", ""], 0, 3, "cpu")
    separate = [batched_scores(model, tokens, [candidate], 0, 3, "cpu")[0] for candidate in ("12", "2", "")]
    assert finished == separate
    assert [n for _, n in finished] == [3, 2, 1]
    prefixes = batched_scores(model, tokens, ["12", ""], 0, 3, "cpu", add_eos=False)
    assert prefixes[1] == (0.0, 0)
    assert prefixes[0][0] > finished[0][0]
