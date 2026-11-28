"""Persistent native decoder for offline batch-one pipeline measurements."""
from base64 import b64decode
import argparse
import json
import sys
import time

import numpy as np

from decode_worker import NativeDecoder, load_lexicon, log_softmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--kenlm-model")
    parser.add_argument("--lm-weight", type=float)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--beam-threshold", type=float, default=25)
    args = parser.parse_args()
    started = time.perf_counter()
    decoder = NativeDecoder(load_lexicon(args.lexicon), beam=args.beam, nbest=args.nbest,
                            beam_threshold=args.beam_threshold, kenlm_model=args.kenlm_model,
                            lm_weight=args.lm_weight)
    print(json.dumps({"mode": decoder.mode, "native_setup_s": time.perf_counter() - started}), flush=True)
    active_trial = None
    for line in sys.stdin:
        request = json.loads(line)
        logits = np.frombuffer(b64decode(request["logits"], validate=True), dtype="<f4")
        if logits.size != 41 * request["output_length"]:
            raise ValueError("Emission frame count does not match the array")
        started = time.perf_counter()
        action = request.get("stream_action")
        if action is None:
            if active_trial is not None:
                raise ValueError("Finish active stream before decoding another trial")
            candidates = decoder.decode(logits.reshape(-1, 41))
        else:
            trial = request["trial_id"]
            if action == "begin":
                if active_trial is not None or logits.size:
                    raise ValueError("Expected begin on an idle decoder with no frames")
                decoder.decoder.decode_begin()
                active_trial, candidates = trial, []
            elif trial != active_trial or action not in {"step", "end"}:
                raise ValueError("Invalid stream action or mismatched trial")
            else:
                if action == "step":
                    emissions = log_softmax(logits.reshape(-1, 41))
                    decoder.decoder.decode_step(emissions.ctypes.data, *emissions.shape)
                    results = [decoder.decoder.get_best_hypothesis(0)]
                else:
                    if logits.size:
                        raise ValueError("End takes no new frames")
                    decoder.decoder.decode_end()
                    results = decoder.decoder.get_all_final_hypothesis()
                    active_trial = None
                candidates = []
                for result in results:
                    words = [int(w) for w in result.words if w >= 0]
                    if 0 not in words and np.isfinite(result.score):
                        text = " ".join(decoder.word_dictionary.get_entry(w) for w in words)
                        candidates.append({"text": text, "score": float(result.score)})
                candidates.sort(key=lambda r: (-r["score"], r["text"]))
                candidates = candidates[:1]
        print(json.dumps({"trial_id": request["trial_id"], "candidates": candidates,
                          "search_s": time.perf_counter() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
