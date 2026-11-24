# Project state

## Accuracy and efficiency continuation

The [continuation registration](CONTINUATION.md) defines the separate v2 study. The v1 report and study freeze remain archived at public revision `fe5fb4081f2ce256b41ba2a91f104a6a64a53deb` and private freeze `c7b12685c9552344144b0f089a0a3c4d7dc6bec88e6df513e52947e926ba7b56`. V1 reporting results informed v2. The old reporting membership serves as an exposed benchmark for any frozen v2 exploratory report.

The v2 scratch tree is `$RCAC_SCRATCH/btt-25-cw/v2`. It reads the original validated logit cache and decoder assets from `workspace` and writes its own runs. A tuning-only replay recovered 1,234 / 6,033 selected word errors and a 986 / 6,033 oracle for the existing compact lists. Separately, the offline laptop ZeroLM smoke benchmark completed 576 real feature-to-text requests over three interleaved passes. That benchmark used the Windows lexical decoder because the cluster KenLM build is Linux.

The original and repaired Flashlight arms completed in `12523601` and `12523602`. Each saved 1,426 predictions without references after a matching pilot. The repaired beam-250 row has 953 / 6,033 tuning edits and an oracle of 566 / 6,033 at up to 50 candidates. The native repair retained all 134,711 pronunciations. The pinned Qwen download and LightBeam dependency setup finished in `12523569`.

LightBeam `12523580` finished the 24-trial tuning comparison. Periodic fusion made 23/156 edits and tied repaired compact on those trials. Both external four-gram pilots timed out after approximately ten warm minutes and produced no tuning scores. The three neural rerankers and both frozen-GRU adaptation arms completed. Freeze `db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8` and job `12523758` produced the first exploratory v2 report and 1,152 matched complete-request timings. The tuning-selected repaired beam 250 with DistilGPT2 scored 14.80% on `dev_report`, 13.84% on full validation and 285 ms p95.

The Transformer stopped at update 24,987 when the job cgroup exceeded the 12 GB target. The checksum-verified resume is `12527227` with the original 40,000-update endpoint. The independent addendum freeze and timing job is `12527230` after successful resume. Next, check `v2/runs/v2-e1-transformer-12527227/train/run_manifest.json` and `v2/runs/v2-report-12527230`, then export their aggregates if complete. The [continuation runbook](CONTINUATION.md) and [ledger](../runs/ledger.csv) preserve the launch details.
## Archived v1 recovery state

Updated 2026-11-24. Baseline, both frozen studies, resource measurement and test export are complete.

The original checkout remains on `model-training` at `aa2ea094`; modified and untracked checkpoints and logs are preserved. This directory is a new source tree with a separate intended history. No old source file or historical result has been copied into it.

Current milestone: laptop inference, lexicon-only word search, and a sustained calibration pilot are complete. Gilbreth has a verified release, a complete validation cache and measured training admission for both calibration modes.

Completed evidence: all 127 local HDF5 files located (45 training, 41 validation, 41 test files; 12,916,863,318 bytes). Native Windows Python 3.13.5 has PyTorch 2.8.0+cu126 and functioning CUDA. RTX 3080 Laptop GPU has 16,384 MiB VRAM. Host memory and GPU memory are measured independently. Raw paths and machine inventories belong in ignored local artifacts.

Recovered official model ZIP from pinned independent Hugging Face mirror; full SHA256 matches official Dryad metadata (`04ab2275...bb4111`). Checkpoint SHA256 `47969ef1de6ba1510e685a02fadca27b7f40aac0b9fc033c8f2e9e0cec8933d1`. Strict loading removes only the uniform `_orig_mod.` compile prefix. Local legacy checkpoints are documented separately and have incompatible experimental token/model contracts.

The complete data has 8072/1426/1450 train/val/test trials. Frozen session/block validation memberships: 926 tune, 500 report; split SHA256 `8b7bbaa899ab516c3a1bba6e21ceb58eaf07798ac2d7a591fa14a3ad0a48ffbf`. The v1 studies kept reporting references closed until their joint freeze. The inherited official checkpoint was selected using official validation; it cannot establish an untouched historical benchmark or unseen-day performance.

Decoder setup: CPython 3.12 plus the Flashlight Text 0.0.7 Windows wheel works. The Windows build lacks the KenLM DLL, so the laptop word run explicitly uses ZeroLM. It saved 1426 / 1426 trials with zero decoder errors in 67.5 s, peak RSS 0.279 GB. At `dev_tune`, 4572 / 6033 word edits give 75.78% WER. The A100 environment uses the Linux wheel and a separately built, pinned KenLM 0.3.0 estimator.

Completed real inference: `runs/baseline-val-fp32` contains all 1426 validation logit arrays and phoneme predictions, produced without references in 132.47 s. `runs/baseline-tune-phones-fp32` scores 926 tuning trials: 2669 / 26680 phoneme edits (10.0037% PER). This is not WER. Real short/median/long logits match the upstream method; padding maximum absolute difference is 7.63e-6 with TF32 disabled. See `assets/local/parity.json`.

Pilot attempts are preserved. `pilot-session-bias` failed on the cuDNN eval/backward requirement, which is fixed and regression-tested. `pilot-session-bias-v2` stopped at 735.21 s when system available memory dipped to 1.989 GB. `pilot-session-bias-v3` completed 900.80 sustained seconds and 7340 steps with actual long trials, validation and checkpoint I/O. Peak RSS was 2.39 GB, private bytes 5.24 GB, and total device use 3.13 GB. Minimum available system RAM was 4.41 GB. See `reports/laptop_baseline.json` for the measured evidence.

Gilbreth verified the matched 127-file release on private scratch and confirmed the frozen split hash there. Full reference-free A100 validation inference saved all 1426 predictions in 81.9 s. Token outputs and frame counts matched the laptop across all trials. KenLM built successfully against Anaconda Boost after an incomplete Boost module caused the first build to fail. The first compact attempt stopped at preflight because its test harness assumed laptop RAM limits under Slurm. Tests now cover both policies. No production decoder weights or candidates were selected on the failed attempt.

Both A100 pilots exceeded 900 sustained seconds with long training trials, tuning-only validation, and checkpoint writes. Head-only reached 14,569 steps with 1.31 GB peak process-tree RSS and 2.03 GB peak device use. Head plus session bias reached 12,334 steps with 1.38 GB RSS and 2.46 GB device use. These are throughput diagnostics.

The three-gram fit used 8,072 official training transcripts. All three compact weights saved 1,426 validation hypotheses. Weight 2.0 was selected with 1,234 edits over 6,033 reference words (20.45% WER) on `dev_tune`. Linux native preflight passed all 23 tests. The trie audit confirmed that the six-label limit drops 224 pronunciation entries and removes all paths for 208 word types.

Confidence tuning selected lambda 1.0. The fixed-candidate oracle was 16.34% WER, 4.11 points below the 20.45% compact WER on `dev_tune`. The selected scorer reduced edits in both grouped tuning subsets, so the original confidence comparison proceeds. Both calibration conditions completed 720 updates and full validation decoding. They tied at 1,230 word edits (20.39% WER) and 2,666 phoneme edits, so the session-bias condition did not warrant a second seed on tuning evidence.

The joint freeze was recorded before opening reporting references. All eight conditions have complete scores for the 500-trial held-back selection and 1,426-trial official validation. See [results.md](../reports/results.md) for the paired changes and resource evidence. The 1,450 exported test rows match the released order and cached hypotheses. They have no hidden-test score.

The private scratch workspace holds the full prediction, metric, logit, and checkpoint evidence. The ignored local mirror holds scored outputs, prediction files, test CSV and verified calibration checkpoint backups. The public results contain aggregate metrics and fixed model-only examples.

Active processes: none. To reproduce a new run, follow [REPRODUCE.md](REPRODUCE.md) and use a new output directory with the verified assets.
