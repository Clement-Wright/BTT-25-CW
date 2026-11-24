"""Measure which pronunciations the installed Flashlight trie retains."""

import argparse
from importlib.metadata import version
from pathlib import Path

from flashlight.lib.text.decoder import Trie

from decode_worker import TOKENS, load_lexicon, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    entries = load_lexicon(args.lexicon)
    trie = Trie(41, 40)
    retained = set()
    effective = []
    count = sum(map(len, entries.values()))
    # The native decoder inserts in sorted word and phone order. Trie label
    # retention follows insertion order even when KenLM scores differ.
    for word_id, (word, variants) in enumerate(sorted(entries.items()), start=1):
        for phones in variants:
            if word_id in trie.insert(list(phones), word_id, 0.0).labels:
                retained.add(word)
                effective.append(f"{word} {' '.join(TOKENS[i] for i in phones)}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = args.output_dir / "searchable_lexicon.txt"
    lexicon.write_bytes("".join(effective).encode("utf-8"))
    write_json(args.output_dir / "trie_audit.json", {
        "flashlight_version": version("flashlight-text"),
        "lexicon_sha256": sha256(args.lexicon),
        "searchable_lexicon_sha256": sha256(lexicon),
        "dictionary_words": len(entries),
        "searchable_words": len(retained),
        "dictionary_pronunciations": count,
        "searchable_pronunciations": len(effective),
        "references_loaded": False,
    })


if __name__ == "__main__":
    main()
