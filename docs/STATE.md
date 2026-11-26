# Project state

No campaign jobs are active. The admitted v2 comparisons and reports are complete. The 10% WER target remains open. The next experiment is to trace native service liveness and IPC around the ten-minute stalls in both external four-gram pilots, then measure candidate coverage on `dev_tune`. [Runbook](CONTINUATION.md) · [Job ledger](../runs/ledger.csv)

## Completed runs

- The v2 tree at `$RCAC_SCRATCH/btt-25-cw/v2` reuses the verified v1 logit cache and decoder assets in `workspace` and writes separate run directories. Replaying the archived compact lists on tuning gave 1,234 selected edits and 986 oracle edits over 6,033 words. The Windows ZeroLM benchmark completed 576 offline feature-to-text requests in three interleaved passes using the laptop lexical decoder. The KenLM build used on the cluster runs on Linux.
- The original and repaired Flashlight jobs `12523601` and `12523602` each saved 1,426 reference-free predictions after their pilots. The repair retained all 134,711 pronunciations. At repaired beam 250, tuning produced 953 selected edits and 566 oracle edits over 6,033 reference words, with up to 50 candidates. Pinned Qwen acquisition and LightBeam setup finished in `12523569`.
- LightBeam periodic fusion tied repaired compact at 23/156 edits on the 24 tuning trials in job `12523580`. Neither four-gram pilot reached admission or produced a tuning WER. All three neural rerankers and both frozen-GRU adaptation arms completed.
- Report job `12523758`, freeze `db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8`, completed 1,152 matched requests. The tuning-selected repaired beam 250 with DistilGPT2 scored 14.80% WER on `dev_report` and 13.84% on full validation, with 285 ms feature-to-text p95. The reporting benchmark had already informed v2 through v1, so the scores are exploratory. [Measured results](../reports/continuation.md)
- The first Transformer attempt stopped at update 24,987. Job `12527227` resumed the checkpoint and reached the registered 40,000 updates using scoped private file-cache advice under the same limits. Addendum job `12527230`, freeze `38b64da860fce6a1369136adbf6c2b9e5878bdaab83952e4b171baa4b6cf2e0a`, measured 56.10% exploratory `dev_report` WER and 110 ms p95.
- Hidden-test export job `12527243` saved all 1,450 test predictions in official order for the original tuning-selected v2 system. The CSV stays on scratch at `v2/runs/v2-test-12527243/submission.csv`. It has no external test score.

## V1 recovery archive

The public v1 results are preserved at `fe5fb4081f2ce256b41ba2a91f104a6a64a53deb`. The v1 private freeze remains `c7b12685c9552344144b0f089a0a3c4d7dc6bec88e6df513e52947e926ba7b56`. The earlier state log records the recovery, original checkout and checkpoint identities, split hash, failed pilots, calibration runs, and first test export. [V1 report](../reports/results.md) · [Historical state at `080134a`](https://github.com/Clement-Wright/BTT-25-CW/blob/080134a/docs/STATE.md)
