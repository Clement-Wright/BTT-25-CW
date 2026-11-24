"""Persistent native decoder for offline batch-one pipeline measurements."""
from base64 import b64decode
import argparse
import json
import sys
import time

import numpy as np

from decode_worker import NativeDecoder, load_lexicon


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
    for line in sys.stdin:
        request = json.loads(line)
        logits = np.frombuffer(b64decode(request["logits"], validate=True), dtype="<f4")
        if logits.size != 41 * request["output_length"]:
            raise ValueError("Emission frame count does not match the array")
        started = time.perf_counter()
        candidates = decoder.decode(logits.reshape(-1, 41))
        print(json.dumps({"trial_id": request["trial_id"], "candidates": candidates,
                          "search_s": time.perf_counter() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
