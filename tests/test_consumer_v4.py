import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from btt25.consumer_replay import early_commit_audit, stable_latencies
from btt25.benchmark import native_decoder
from btt25.model import OfficialGRU, gaussian_kernel, smooth_valid
from btt25.online_encoder import OnlineGRU, OnlineTwoRate
from btt25.v4_rate import RateEncoder


def test_streaming_valid_smoothing_and_patching_match_offline():
    torch.manual_seed(19)
    torch.set_num_threads(1)
    model = OfficialGRU(4, 8, 1, 41, n_layers=2).eval()
    kernel = gaussian_kernel()
    x = torch.randn(55, 4)
    online = OnlineGRU(model, kernel, 0)
    with torch.no_grad():
        assert online.push(x[:21]) is None
        first = online.push(x[21:22])
        assert first.shape == (1, 41)
        assert online.push(x[22:25]) is None
        second = online.push(x[25:26])
        rest = online.push(x[26:])
        collected = torch.cat([first, second, rest])
        batch = model(smooth_valid(x[None], kernel), [0])[0]
        torch.testing.assert_close(collected, batch, rtol=1e-5, atol=1e-6)
        changed = x.clone()
        changed[26:] += 12
        perturbed = model(smooth_valid(changed[None], kernel), [0])[0]
        torch.testing.assert_close(collected[:2], perturbed[:2])
        assert online.raw.shape[1] <= 8 and online.patch.shape[1] <= 13
    with pytest.raises(ValueError, match="intracortical"):
        OnlineGRU(model, kernel, 0, modality="scalp EEG")


def test_stability_requires_every_subsequent_prefix_to_agree():
    drafts = [(1, ["a"]), (2, ["b"]), (3, ["a", "d"]), (4, ["a"])]
    assert stable_latencies(drafts, ["a", "d"], 5) == [3, 5]


def test_online_two_rate_uses_new_bins_without_extra_gru_updates():
    torch.manual_seed(23)
    torch.set_num_threads(1)
    m = RateEncoder(OfficialGRU(4, 8, 1, 41, n_layers=2), "two_rate", gaussian_kernel()).eval()
    with torch.no_grad():
        m.mid[-1].weight.normal_(std=.1)
        bins = torch.randn(63,4)
        online = OnlineTwoRate(m,0)
        assert online.push(bins[:21]) is None
        first = online.push(bins[21:22])
        hidden = online.hidden.clone()
        second = online.push(bins[22:24])
        torch.testing.assert_close(hidden, online.hidden, rtol=0, atol=0)
        remaining = online.push(bins[24:])
        expected = m(bins[None],0)[0]
        torch.testing.assert_close(torch.cat((first,second,remaining)),expected,rtol=1e-5,atol=1e-6)


def test_early_confirmation_requires_stable_following_word_and_counts_disagreements():
    drafts = [(i, ["a", "b"]) for i in range(5)] + [(5, ["a", "c"]), (6, ["x", "d"])]
    audit = early_commit_audit(drafts, ["x", "d"])
    assert audit["words_confirmed_early"] == 1
    assert audit["first_confirmation_s"] == 4
    assert audit["later_drafts_conflicting"] == 1
    assert not audit["final_prefix_agreement"]


def test_native_stream_preserves_an_unambiguous_word_path(tmp_path):
    python = Path(os.getenv("BTT_DECODER_PYTHON", ".decoder-venv/Scripts/python.exe")).resolve()
    if not python.is_file():
        pytest.skip("Set BTT_DECODER_PYTHON to the optional native decoder environment")
    lexicon = tmp_path / "phones.txt"
    lexicon.write_text("alpha AA\nbeta AE\n")
    args = Namespace(
        decoder_python=python, lexicon=lexicon, nbest=1, beam_threshold=25,
        kenlm_model=None, start_timeout=30, request_timeout=5,
    )
    logits = np.full((6, 41), -20.0, dtype=np.float32)
    logits[np.arange(6), [1, 40, 0, 2, 40, 0]] = 20.0
    with native_decoder(args, 50, tmp_path / "native.log") as (decode, _):
        assert decode("stream-test", logits)["candidates"][0]["text"] == "alpha beta"
        decode("stream-test", logits[:0], "begin")
        for i in range(len(logits)):
            decode("stream-test", logits[i : i + 1], "step")
        assert decode("stream-test", logits[:0], "end")["candidates"][0]["text"] == "alpha beta"
