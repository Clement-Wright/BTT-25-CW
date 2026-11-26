# BTT-25-CW

BTT-25-CW decodes attempted speech from the released T15 intracortical recordings. It runs the authors’ pretrained GRU on 20 ms feature bins, saves the phoneme logits, and uses those logits to compare word decoders without rerunning the encoder. The recordings come from one participant. [Dataset and terms](docs/DATA_CARD.md)

## How well it works

| System | Word errors on `dev_report` | WER | Feature-to-text p95 |
| --- | ---: | ---: | ---: |
| Original compact decoder | 678 / 3,412 | 19.87% | 104 ms |
| Repaired beam-250 search | 571 / 3,412 | 16.74% | 250 ms |
| Repaired search + DistilGPT2 | 505 / 3,412 | **14.80%** | 285 ms |

![Three decoder configurations and one encoder control, with report error rates on the left and matched latency percentiles on the right](reports/figures/v2_accuracy_latency.png)

DistilGPT2 had the lowest WER on tuning and on full validation, where it reached **13.84% WER** (1,307 errors in 9,445 words). Qwen3-4B made two fewer errors on `dev_report`. Treat both v2 scores as exploratory. We had already seen the v1 report when we planned these runs. Latency comes from three interleaved passes over the same 96 requests per system on one A100. [V2 report and measurements](reports/continuation.md)

The largest gain came from search. Flashlight’s six-label terminal limit made 208 dictionary words unreachable. After removing it and widening the beam, the best available candidate per trial would score **9.38% WER** on tuning. The decoder still often ranks a worse candidate first. [Candidate coverage](reports/continuation.md#candidate-coverage)

The hidden-test export has 1,450 ordered predictions. The release withholds test labels, so we can report coverage only. [Export instructions](docs/CONTINUATION.md)

## Run it

Get the authorized T15 release and verify its hashes. Use [the setup guide](docs/REPRODUCE.md) to prepare `configs/local.yaml`, then run:

```bash
python -m btt25.cli infer --config configs/local.yaml --split val --run-dir runs/baseline-val
python -m btt25.cli evaluate --config configs/local.yaml \
  --predictions runs/baseline-val/predictions.jsonl \
  --output-dir runs/baseline-phones-tune --phoneme-only
```

The second command scores phonemes on `dev_tune` by default. For word decoding, checkpoints, or reproduction on an A100, follow the [Gilbreth instructions](docs/GILBRETH.md) and [v2 commands](docs/CONTINUATION.md). Keep data, weights, and reference-bearing outputs outside Git.

## Reports and reference

- Beam search, Qwen, LightBeam, the Transformer, and the diphone comparison. [V2 report](reports/continuation.md)
- Confidence gating and session calibration. [V1 report](reports/results.md) · [Archived v1 revision](https://github.com/Clement-Wright/BTT-25-CW/tree/fe5fb4081f2ce256b41ba2a91f104a6a64a53deb)
- Weights, preprocessing, and licenses. [Model card](docs/MODEL_CARD.md) · [Provenance](docs/PROVENANCE.md) · [Third-party terms](THIRD_PARTY_NOTICES.md)
- Why we chose these comparisons and where work stopped. [Research context](docs/RESEARCH_CONTEXT.md) · [Recovery audit](docs/RECOVERY_AUDIT.md) · [Project state](docs/STATE.md)
