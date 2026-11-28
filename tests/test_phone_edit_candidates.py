from math import log

import pytest

from btt25.data import TOKENS
from btt25.phone_edit_candidates import Neighbors, Trigram, proposals


def test_trigram_backs_off_after_mapping_unknown_word(tmp_path):
    path = tmp_path / "trial.arpa"
    path.write_text("\\1-grams:\n-1 <unk>\n0 <s> -0.2\n-0.4 </s>\n-1 cat -0.1\n"
                    "\\2-grams:\n-0.5 <s> cat -0.3\n-0.2 cat </s>\n\\end\\\n")
    lm = Trigram(path)
    assert lm.sentence(["cat"]) == pytest.approx((-0.5 - 0.3 - 0.2) * log(10))
    assert lm.sentence(["absent"]) == pytest.approx((-0.2 - 1 - 0.4) * log(10))


def test_neighbors_keep_homophones_and_require_a_true_single_edit(tmp_path):
    p = tmp_path / "lexicon"
    p.write_text("tail T EY L\ntale T EY L\ndale D EY L\nail EY L\ncattail K T EY L\n"
                 "rotated EY L T\nfar F AA R\n")
    cfg = {"word_alternatives":16,"other_edit_cost":1.,"voicing_substitution_cost":.5,
           "sil_id":40,"proposal_edit_penalty_nats":4.,"beam_width":32,"new_sentence_limit":25}
    index = Neighbors(p,{},cfg)
    segment = tuple(TOKENS.index(phone) for phone in ("T","EY","L"))
    costs = {word:cost for cost,_,word,_ in index.lookup(segment)}
    assert costs == {"tail":0., "tale":0., "dale":.5, "ail":1., "cattail":1.}

    class FlatLM:
        @staticmethod
        def step(context, word):
            return (context+(word,))[-2:], 0.

    choices = dict(proposals((40,*segment,40,40), index, FlatLM(), cfg))
    assert choices["tale"] == (40,*segment,40,40)
