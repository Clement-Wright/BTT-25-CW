import itertools

import numpy as np

from btt25.ctc import greedy_ctc
from btt25.evaluation import edit_counts
from btt25.phone_robustness import edit_distances, mbr, path_scores
from btt25.strong_rate_decode import score_candidates
from btt25.phone_word_fusion import viterbi_scores


def test_word_distance_including_empty_repeated_and_long_sentences():
    rng = np.random.default_rng(13)
    words = [list("aab"), [], ["two words"], list("aba"), list("bbb")]
    words.extend(rng.choice(list("abcd"), 64 + i).tolist() for i in range(3))
    distances = edit_distances(words)
    expected = [[edit_counts(a, b)["errors"] for b in words] for a in words]
    np.testing.assert_array_equal(distances, expected)


def test_ctc_sum_includes_repeated_phones_and_blank_only_hypothesis():
    logits = np.array([[.3, .1, 0.], [0., .6, .1], [.7, .2, .4]], dtype=np.float32)
    paths = [[1, 1], [1], [], [2, 1], [1]]
    p = np.exp(logits.astype(float))
    p /= p.sum(1, keepdims=True)
    reference = [sum(np.prod(p[np.arange(3), seq])
                     for seq in itertools.product(range(3), repeat=3) if greedy_ctc(seq) == path)
                 for path in paths]
    np.testing.assert_allclose(path_scores(logits, paths), np.log(reference))
    best = [max(np.prod(p[np.arange(3), seq]) for seq in itertools.product(range(3), repeat=3)
                if greedy_ctc(seq) == path) for path in paths]
    np.testing.assert_allclose(viterbi_scores(logits,paths),np.log(best))


def test_expected_word_edits_can_choose_non_map_sentence():
    words = [list("xyz"), list("abc"), list("abd"), list("aec")]
    scores = np.log([.34, .30, .19, .17])
    assert mbr(scores, edit_distances(words), 1.) == 1


def test_deployed_ranker_reuses_one_score_batch_and_matches_experimental_mass(monkeypatch):
    calls = []

    class InjectedScorer:
        device = object()

        def score_many(self, texts, batch_size):
            calls.append((texts, batch_size))
            return [{"log_likelihood": -1., "token_count": 2} for text in texts]

    monkeypatch.setattr("torch.cuda.synchronize", lambda device: None)
    logits = np.array([[.3,.1,0.],[0.,.6,.1],[.7,.2,.4]],dtype=np.float32)
    paths = [[1,1],[1]]
    choices = [{"text":str(i),"phoneme_ids":seq,"component_scores":{
               "acoustic":-i,"kenlm_contribution":0.,"rank_score":-i}} for i,seq in enumerate(paths)]
    result = score_candidates(choices, logits, InjectedScorer(), {"candidate_limit":25,"weight":1.25,"batch_size":8})
    assert calls == [(["0","1"],8)]
    assert result["text"] == "0" and result["aligned_text"] == "1"
    received = sorted(result["scores"],key=lambda row:row["source_rank"])
    np.testing.assert_allclose([r["ctc_log_mass"] for r in received],path_scores(logits,paths))
