# Recovery audit

Audit date: 16 November 2026. This document records inspected source, checkpoint metadata and recovery checks. Current experiment status belongs in [STATE.md](STATE.md). Private inventories retain local paths and detailed evidence; this public account includes no participant transcript examples or borrowed source.

The recovered checkout was on `model-training` at `aa2ea09415402c364f235069fa5ed59fe1220e5c`, seven commits behind its cached branch tip. The earlier handoff inspected other private revisions. The audit read current files and cached history without checkout, reset, merge or fetch, preserving modified/untracked checkpoints and loss logs. Historical findings were reconciled against this checkout: an evaluator now exists, and the trainer's child-layer reset is commented out. Describing the old default as the current source would be inaccurate.

Local checkpoint metadata shows two different experimental model families:

| Asset | Inspected contract | Evidence limit |
|---|---|---|
| `latest.pth` | 512 inputs; two bidirectional 256-unit GRU layers; LayerNorm; 759 diphone/41 phoneme outputs; saved epoch 8, step 4,212 | Structurally resembles the current experimental default. Vocabulary is null, extra metadata is empty, and source/split/preprocessing identity is missing. |
| `best.pth` | 512 inputs; one bidirectional 1,024-unit GRU layer; no LayerNorm; 759 diphone/41 phoneme outputs; saved epoch 149, step 101,700 | Resembles an older single-layer family, including revision `30d1ed1`. It cannot strictly load into the current two-layer default. |

The original audit read restricted metadata records, not tensor storage. Shapes, filenames and saved counters establish neither successful training nor accuracy. A matching historical loss trace supports an experiment association, without proving checkpoint provenance. Neither asset satisfies the official T15 architecture, ordered-session and token contracts.

The following defects were verified in the recovered checkout. Paths and line numbers in this table refer to that historical source, not the new implementation.

| Defect | Concrete evidence and consequence |
|---|---|
| Validation selected from training | `model_handling/rnn_trainer.py:539-540` independently samples training and validation files from `train_data`; it ignores `val_data` and does not exclude overlap. This direct-script path cannot establish held-out validation. |
| Resume updates the wrong model | `RNNTrainer.load_checkpoint` constructs a replacement model while the existing optimizer remains bound to the previous parameters. The training caller does not rebuild that binding or apply the returned progress counters. |
| Diphone schedule never activates during normal training | `_step` requires phase 3 at line 275, while the normal training path sets that phase only after training at line 413. The configured auxiliary objective is therefore inactive in that path. |
| Blank, AA and padding disagree | `set_marginal_mask` excludes IDs 0 and 1 although AA is 1. Forward adds a suppressed padding column and derives phoneme blank from diphone index zero, which represents a real transition. The same blank-zero CTC criterion is used for diphones. |
| Lengths and metrics alter the task | `NeuralDataset` reads declared lengths without slicing valid prefixes; the bidirectional model ignores them. Dataset/trainer/collation padding conventions differ. Sequence cleanup removes blank-separated repeats, PER collapses repeated reference phones, and trainer evaluation inspects only the first batch member using positional agreement. Several auxiliary penalties are detached. |
| Evaluator cannot establish a valid baseline | `evaluation/evaluate_model.py` permits `strict=False` loading and random-weight fallback, uses a different preprocessing path, collapses CTC in the wrong order and assigns session-only trial IDs. Its WER loop accesses absent `true_sentence`/`pred_sentence` keys. It also materializes all selected sessions before inference. |
| Entry point and provenance are incomplete | `main.py:98-102` uses placeholder/incompatible constructor and trainer arguments. Checkpoint kwargs omit non-default architecture settings; split, vocabulary, RNG and complete configuration provenance are absent. Loss-only CSVs do not establish a research result. |

The 2024-derived repository and a separate convolutional/bidirectional-GRU folder were also inspected. Neither supplied a verified, compatible T15 weight package. The newer cached 2024 branch differed from its local checkout, so those versions were kept distinct. No root redistribution license was found in the inspected legacy candidates. Their code and example results remain outside this source tree; historical contributors retain credit in [PROVENANCE.md](PROVENANCE.md).

Recovery instead uses the [official July 2025 Dryad release](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) and the [pinned official method](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/980a5b7e96392210edf976d76bdb5d15d2e0e62b). The original neural source has no verified blanket redistribution grant. BTT-25-CW separately implements its method and uses privately acquired upstream files for parity checks. Data/model terms, installed dependencies and pronunciation-dictionary terms remain separate.

Verified recovery evidence includes:

- All 127 released HDF5 session files were located: 45 training, 41 validation and 41 test files, containing 8,072/1,426/1,450 trials. Missing validation files do not renumber the 45-session checkpoint mapping.
- Dryad downloads returned an access error in this environment. The complete model ZIP recovered from a pinned independent mirror matches authoritative Dryad SHA-256 `04ab2275eeb48abad9b4011dde6f7c9e5dd9f643b8a677c1ba5e06dc53bb4111`. The [asset manifest](../assets/manifest.json) identifies the mirror, raw checkpoint and arguments separately.
- Strict loading validates checkpoint, tokens, ordered days and preprocessing, removing only the documented uniform compile prefix. Unknown weights fail; random initialization cannot stand in for a trained baseline.
- Real short, median and long trials matched upstream smoothing and logits exactly in the recorded FP32 parity check. With TF32 disabled, the padded-batch valid-prefix maximum absolute difference was `7.62939453125e-6`. No references were loaded for parity.
- Feature-only inference produced all 1,426 validation predictions. The separate 926-trial tuning score has 2,669 edits over 26,680 reference phones: 10.0037% PER. Word error rate was unmeasured in that phoneme-only result. Reporting labels were excluded.
- CPU recovery regressions verify that fresh training rejects existing evidence, interrupted attempts preserve prior metadata/resources, and a three-update interrupted/resumed run matches uninterrupted model and Adam tensors exactly. This is a small synthetic recovery check, not a completed GPU adaptation study.

Failures remain evidence. The first session-bias pilot exposed cuDNN's evaluation-mode backward restriction; calibration now requests backward buffers while explicitly disabling recurrent dropout. A later attempt stopped at 735.21 seconds when system available memory fell below 2 GB, despite process peak RSS of 2.401 GB. Its checkpoint was retained, and it does not qualify as the required sustained 15-minute pilot. Host process-tree RAM and independent VRAM are measured separately; a machine with more installed RAM does not relax the project's 12 GB target/16 GB ceiling.

The recovery checks above do not establish complete word decoding, a hidden-test score or an improvement from either primary study. Remaining runs and their failures must stay visible in [STATE.md](STATE.md), with frozen split/configuration identities and the original baseline preserved.
