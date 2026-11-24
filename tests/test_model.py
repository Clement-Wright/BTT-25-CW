import pytest
import torch
from btt25.model import OfficialGRU, gaussian_kernel, smooth_valid, output_length, load_encoder, unwrap_compiled_state


def small_model():
    return OfficialGRU(neural_dim=512, n_units=8, n_days=2, n_classes=41, n_layers=2, patch_size=14, patch_stride=4).eval()


def test_real_smoothing_support_and_lengths():
    kernel = gaussian_kernel(2, 100)
    assert len(kernel) == 9
    assert float(kernel.sum()) == pytest.approx(1)
    x = torch.randn(2, 80, 512)
    smoothed = smooth_valid(x, kernel)
    assert smoothed.shape == (2, 72, 512)
    model = small_model()
    assert model(smoothed, torch.tensor([0, 1])).shape[1] == output_length(80, len(kernel), 14, 4) == 15
    with pytest.raises(ValueError):
        output_length(20, 9, 14, 4)


def test_padding_does_not_change_valid_logits():
    torch.manual_seed(42)
    model = small_model()
    short, long = torch.randn(1, 81, 512), torch.randn(1, 131, 512)
    k = gaussian_kernel()
    alone = model(smooth_valid(short, k), torch.tensor([0]))
    padded = torch.zeros(2, 131, 512)
    padded[0, :81], padded[1] = short[0], long[0]
    together = model(smooth_valid(padded, k), torch.tensor([0, 1]))
    torch.testing.assert_close(alone[0], together[0, :alone.shape[1]], atol=1e-6, rtol=1e-5)


def test_unknown_day_rejected():
    model = small_model()
    with pytest.raises(ValueError, match="day"):
        model(torch.randn(1, 30, 512), torch.tensor([2]))


def test_missing_checkpoint_never_initializes_random(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_encoder(tmp_path / "missing", tmp_path / "args.yaml")


def test_explicit_compile_wrapper_removal_and_mixed_rejection():
    state = {"_orig_mod.out.weight": torch.ones(2, 3), "_orig_mod.h0": torch.zeros(1, 1, 3)}
    assert set(unwrap_compiled_state(state)) == {"out.weight", "h0"}
    with pytest.raises(ValueError, match="Mixed"):
        unwrap_compiled_state({**state, "out.bias": torch.zeros(2)})
