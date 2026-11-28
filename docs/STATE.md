# Project state

## T15 output rate and input representation study

The 13 trained conditions finished 2,048 matched training updates each. All have checkpoints and 926 saved tuning predictions. Job `12625888` timed 1,008 complete feature-to-text requests across 14 systems on one A100-40GB. The official checkpoint still has the lowest PER. The best word decoder on those emissions made 764 word errors in 6,033 tuning words (12.66%). Official test export `12626394` saved 1,450 nonempty predictions in order. The hidden-test score is unknown. [Encoder results](../reports/rate_v4.md) · [Word decoder](../reports/phone_robustness.md) · [Job ledger](../runs/ledger.csv)

Eight rate-aware decoding conditions saved all 1,426 validation predictions apiece in `12625921`. The initial scoring step stopped on a lexical replay difference. Job `12627501` checked the emission hashes, documented the differences and scored the saved outputs against the 926 tuning IDs. Merging agreeing pairs reduced the two-rate model's beam-50 word errors from 1,719 to 1,192. A laptop CPU replay delivered one new posterior every 40 ms after the initial input window. [Rate-aware decoder](../reports/rate_decode_v4.md) · [Cadence diagnostics](../reports/cadence_v4.md) · [CPU replay](../reports/consumer_v4.md) · [Separate scalp EEG experiment](../reports/scalp_eeg.md)

Private per-trial evidence and training artifacts are under `$RCAC_SCRATCH/btt-25-cw/v4-rate-20261127/runs`. Direct 10 ms acquisition would require finer native recordings from the same trials. Prospective word-accuracy validation requires eligible new, independently scored transcripts. [Study design](RATE_STUDY_V4.md) · [Gilbreth commands](REPRODUCE_RATE_STUDY.md) · [Laptop commands](REPRODUCE_LAPTOP_V4.md)

The final job `12627738` finished in 5 min 55 s. Under the same expanded word decoder the two-rate checkpoint made 1,152 errors with all posteriors and 914 (15.15%) with agreeing pairs merged. All 144 warm timing requests reproduced the saved texts. No campaign jobs remain active. Thirteen final checkpoints, including optimizer state, were copied to private storage and verified against both their source hashes and completed training receipts. [Paired results](../reports/two_rate_expanded.md)

The next accuracy comparison would fit a ranker on out-of-fold training predictions, keep the 764-edit decoder as its control, and reserve an eligible evaluation set before choosing its weights. A public integrity manifest covers the completed registrations and aggregate reports. On a checkout with LF line endings, run `sha256sum -c reports/v4-release.sha256` at the repository root. [What the candidate lists can recover](../reports/phone_robustness.md#expanded-lists-with-distilgpt2)

## Earlier temporal feature study (v3)

Jobs `12618760` and `12618761` completed all eight frozen-GRU conditions on the same 926 tuning trials. Official smoothing at 80 ms output steps scored 17.72% WER and 10.00% PER with repaired beam-50 decoding. Five-bin averaging scored 17.42% WER, with a paired interval spanning zero. Shortening the output step to 40 or 20 ms degraded WER to 30.08% or 73.53%. Each condition saved predictions, counts, and resources. The campaign checkout and artifacts are at `$RCAC_SCRATCH/btt-25-cw/v3-smoothing-20261127`. [Temporal screen](../reports/temporal.md)

Reference-phone job `12619225` completed on all 926 tuning trials. Repaired beam-250 Flashlight and KenLM scored 15.80% WER from saved GRU emissions and 2.34% from one-frame-per-phone reference emissions, retaining the exact phone path on 922 trials. Exact CMU lookup with reference SIL and a training unigram scored 2.69%. The injected emissions use label-derived phones, boundaries, and confidence. The earlier sustained-phone construction yielded zero exact paths.

All three 900-second training pilots finished and saved predictions for all 926 tuning trials at both checkpoints. At 256 updates, the head-only control `12619117` scored 17.87% WER, channel-wise fusion `12619118` scored 17.95%, and dense fusion `12619119` scored 17.85%. All three scored above the unchanged checkpoint's 17.72% WER at both endpoints.

Setup job `12619071` finished in 18 minutes 37 seconds and verified both requested Q4_K_M weights. Jobs `12620640` and `12620647` loaded all weights onto their A100s, then stopped when serial and batched fixture likelihoods missed the tolerance. Jobs `12620737` and `12620738` log both scores, check for state leaking across candidates, and use serial teacher forcing if the difference exceeds the tolerance. Check `sacct -j 12620737,12620738 --format=JobID,State,Elapsed,ExitCode,MaxRSS` and each run's `metrics.json`, `timing.jsonl`, and `backend.log`.

Qwen3.6 job `12620738` finished all 926 tuning trials at 13.92% WER. Qwen3.8 pilot `12620737` saved all 24 pilot predictions and stopped when its conservative forecast for the remaining work was 7,792 seconds. Continuation `12620905` rejected the hash of a freshly linked scorer. Job `12621131` reused the exact pilot executable, reproduced the score of a pilot trial, and finished all 926 tuning trials at 14.02% WER. DistilGPT2 scored 13.29% on the same candidate lists. Scorer artifacts and scheduler receipts are on scratch and the local ignored `runs/v3-temporal-screen` directory.

Reference-free job `12621644` finished 144 requests on one A100-SXM4 over three alternating passes of the same 24 cached-candidate requests. P95 was 2.66 s for Qwen3.6 and 6.98 s for Qwen3.8. The complete feature-to-text measurements remain in the v2 benchmark.

The v3 study motivated the trained 40 ms comparison above. [Temporal report](../reports/temporal.md) · [Runbook](CONTINUATION.md)

## Completed runs

- The v2 tree at `$RCAC_SCRATCH/btt-25-cw/v2` reuses the verified v1 logit cache and decoder assets in `workspace` and writes separate run directories. Replaying the archived compact lists on tuning gave 1,234 selected edits and 986 oracle edits over 6,033 words. The Windows ZeroLM benchmark completed 576 offline feature-to-text requests in three interleaved passes using the laptop lexical decoder. The KenLM build used on the cluster runs on Linux.
- The original and repaired Flashlight jobs `12523601` and `12523602` each saved 1,426 reference-free predictions after their pilots. The repair retained all 134,711 pronunciations. At repaired beam 250, tuning produced 953 selected edits and 566 oracle edits over 6,033 reference words, with up to 50 candidates. Pinned Qwen acquisition and LightBeam setup finished in `12523569`.
- LightBeam periodic fusion tied repaired compact at 23/156 edits on the 24 tuning trials in job `12523580`. Neither four-gram pilot reached admission or produced a tuning WER. All three neural rerankers and both frozen-GRU adaptation arms completed.
- Report job `12523758`, freeze `db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8`, completed 1,152 matched requests. The tuning-selected repaired beam 250 with DistilGPT2 scored 14.80% WER on `dev_report` and 13.84% on full validation, with 285 ms feature-to-text p95. The reporting benchmark had already informed v2 through v1, so the scores are exploratory. [Measured results](../reports/continuation.md)
- The first Transformer attempt stopped at update 24,987. Job `12527227` resumed the checkpoint and reached the registered 40,000 updates using scoped private file-cache advice under the same limits. Addendum job `12527230`, freeze `38b64da860fce6a1369136adbf6c2b9e5878bdaab83952e4b171baa4b6cf2e0a`, measured 56.10% exploratory `dev_report` WER and 110 ms p95.
- Hidden-test export job `12527243` saved all 1,450 test predictions in official order for the original tuning-selected v2 system. The CSV stays on scratch at `v2/runs/v2-test-12527243/submission.csv`. It has no external test score.

## V1 recovery archive

The public v1 results are preserved at `fe5fb4081f2ce256b41ba2a91f104a6a64a53deb`. The v1 private freeze remains `c7b12685c9552344144b0f089a0a3c4d7dc6bec88e6df513e52947e926ba7b56`. The earlier state log records the recovery, original checkout and checkpoint identities, split hash, failed pilots, calibration runs, and first test export. [V1 report](../reports/results.md) · [Historical state at `080134a`](https://github.com/Clement-Wright/BTT-25-CW/blob/080134a/docs/STATE.md)
