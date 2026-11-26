# Recovery audit

The source and checkpoint audit took place on 16 November 2026. This account was updated on 26 November to distinguish those initial findings from the experiments completed afterward. Private inventories retain local paths and detailed evidence; public reports contain aggregates and selected model outputs without reference labels.

The recovered checkout was on `model-training` at `aa2ea09415402c364f235069fa5ed59fe1220e5c`, seven commits behind its cached branch tip. The handoff had inspected other private revisions. We read the current files and cached history without checkout, reset, merge or fetch, preserving modified and untracked checkpoints and loss logs. Two handoff findings had already changed: an evaluator existed, and the trainer's child-layer reset was commented out. The findings below apply to the inspected checkout.

Checkpoint metadata identified two experimental model families:

| Asset | Inspected contract | Evidence limit |
|---|---|---|
| `latest.pth` | 512 inputs; two bidirectional 256-unit GRU layers; LayerNorm; 759 diphone/41 phoneme outputs; saved epoch 8, step 4,212 | Resembles the current experimental default. Vocabulary is null, extra metadata is empty, and source/split/preprocessing identity is missing. |
| `best.pth` | 512 inputs; one bidirectional 1,024-unit GRU layer; no LayerNorm; 759 diphone/41 phoneme outputs; saved epoch 149, step 101,700 | Resembles an older single-layer family, including revision `30d1ed1`. It cannot strictly load into the two-layer default. |

This inspection read restricted metadata records rather than tensor storage. Shapes, filenames and saved counters do not establish accuracy or a verified training history. A matching loss trace supports an association with an older experiment, but cannot prove checkpoint provenance. Neither asset satisfies the official T15 architecture, ordered-session and token contracts.

The source audit found the following failures. Paths and line numbers refer to the historical checkout, not BTT-25-CW.

| Failure | Evidence and consequence |
|---|---|
| Validation sampled from training | `model_handling/rnn_trainer.py:539-540` independently samples training and validation files from `train_data`, ignores `val_data`, and permits overlap. That direct-script path cannot establish held-out validation. |
| Resume leaves the optimizer on the old model | `RNNTrainer.load_checkpoint` constructs a replacement model, while the optimizer remains bound to the previous parameters. The caller neither rebuilds that binding nor applies the returned progress counters. |
| Normal training never enables the diphone schedule | `_step` requires phase 3 at line 275. The normal path sets that phase only after training at line 413, leaving the configured auxiliary objective inactive. |
| Blank, AA and padding have incompatible meanings | `set_marginal_mask` excludes IDs 0 and 1, although AA is 1. Forward adds a suppressed padding column and derives phoneme blank from diphone index zero, which represents a real transition. Diphone CTC also uses blank zero. |
| Length handling and metrics change the sequences | `NeuralDataset` reads declared lengths without slicing valid prefixes, and the bidirectional model ignores them. Dataset, trainer and collation padding differ. Sequence cleanup removes blank-separated repeats; PER collapses repeated reference phones. Trainer evaluation uses only the first batch member and positional agreement. Several auxiliary penalties are detached. |
| The evaluator cannot verify a baseline | `evaluation/evaluate_model.py` allows `strict=False` loading and random-weight fallback, uses different preprocessing, collapses CTC in the wrong order, and assigns session-only trial IDs. Its WER loop accesses absent `true_sentence`/`pred_sentence` keys. It also materializes all selected sessions before inference. |
| Entry point and provenance are incomplete | `main.py:98-102` passes placeholder/incompatible constructor and trainer arguments. Checkpoint kwargs omit non-default architecture settings. Split, vocabulary, RNG and complete configuration provenance are absent. Loss-only CSVs cannot establish a research result. |

A 2024-derived repository and a separate convolutional/bidirectional-GRU folder were also inspected. Neither supplied a verified compatible T15 weight package. The cached 2024 branch differed from its checkout, so those versions were kept distinct. No root redistribution license was found in the inspected legacy candidates. Their code and example results remain outside this repository; [PROVENANCE.md](PROVENANCE.md) retains the historical credits.

Recovery used the [official July 2025 Dryad release](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) and the [pinned official method](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/980a5b7e96392210edf976d76bdb5d15d2e0e62b). No blanket redistribution grant was verified for the original neural source. The new implementation follows the method, while privately acquired upstream files support parity checks. Data/model terms and external dependency licenses remain separate.

Recovery checks established that:

- All 127 HDF5 session files were present: 45 training, 41 validation and 41 test files, containing 8,072/1,426/1,450 trials. Missing validation session files do not renumber the checkpoint's 45-session map.
- The model ZIP acquired from a pinned independent mirror matched authoritative Dryad SHA-256 `04ab2275eeb48abad9b4011dde6f7c9e5dd9f643b8a677c1ba5e06dc53bb4111`. Direct Dryad downloads had failed in this environment. The [asset manifest](../assets/manifest.json) records the mirror, raw checkpoint and arguments separately.
- Strict loading checks weights, tokens, ordered days and preprocessing, removing only the documented uniform compile prefix. Unknown or incompatible weights fail.
- Real short, median and long trials matched upstream smoothing and logits exactly in the recorded FP32 check. With TF32 disabled, the maximum valid-prefix logit difference under batch padding was `7.62939453125e-6`. Parity loaded no references.
- Feature-only inference saved every validation trial. The first separate tuning score made 2,669 edits over 26,680 phones, or 10.0037% PER, across 926 trials. That initial phoneme-only result had no WER. Later laptop/A100 comparison matched greedy phones and frame counts across all 1,426 trials.
- CPU recovery regressions reject fresh training over existing evidence and preserve interrupted-attempt metadata and resources. A three-update interrupted/resumed fixture matched uninterrupted model and Adam tensors exactly. That synthetic test checks resume correctness, independently of the later GPU studies.

The pilot failures were retained. The first session-bias attempt exposed cuDNN's evaluation-mode backward restriction; calibration now requests backward buffers with recurrent dropout explicitly disabled. Another attempt stopped at 735.21 seconds when available system memory fell below 2 GB, despite peak process RSS of 2.401 GB. Its checkpoint survived, but the attempt did not satisfy sustained admission.

The subsequent laptop pilot completed 900.80 sustained seconds and 7,340 steps with long trials, validation and checkpoint I/O. It peaked at 2.39 GB process-tree RSS and 3.13 GB total device usage, with at least 4.41 GB available system RAM. [Measured laptop evidence](../reports/laptop_baseline.json) keeps the private-memory counter separate. The 12 GB host target and 16 GB ceiling remain independent of installed host RAM and VRAM.

The work continued beyond baseline recovery. Both v1 calibration conditions completed 720 updates; the confidence policies, paired analysis and ordered hidden-test export also completed under their joint freeze. V2 then completed the repaired native search, three fixed-candidate scorers, the 24-trial LightBeam feasibility comparison and matched monophone/diphone adaptation. The scratch-trained Transformer stopped with a valid checkpoint at update 24,987, then resumed to the registered 40,000 updates with scoped file-cache advice under the same memory limits.

Two external four-gram pilots timed out on native requests after about ten warm minutes. Their cgroup peaks were below 1 GB with zero OOM events; neither produced an admitted tuning score. The failed attempts remain part of the evidence rather than being counted as completed accuracy comparisons.

The [v1 report](../reports/results.md) and [v2 continuation](../reports/continuation.md) contain the finished measurements, including harmful results, uncertainty, resource accounting and test coverage. V1 report exposure makes v2 reporting exploratory. Hidden test labels remain unavailable, and the 10% WER target remains unmet. [STATE.md](STATE.md) and the [ledger](../runs/ledger.csv) preserve the campaign status and receipts.
