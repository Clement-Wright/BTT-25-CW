# Model card

The baseline is the official pretrained T15 GRU. Its 44,315,177 parameters comprise 45 day-specific 512-dimensional affine transforms with Softsign, five unidirectional GRU layers of 768 units, a learned initial state and a 41-class phoneme head. The token order is fixed: blank is 0, AA is 1, and SIL/boundary is 40. Padding has no output class.

The loader checks the release checksum, architecture, full token order and ordered day map. The checkpoint has a uniform `_orig_mod.` prefix from `torch.compile`; only that documented wrapper is removed before strict loading. A narrow restricted-loading allowlist handles the original NumPy scalar metadata. Missing keys, incompatible shapes or unknown weights raise errors. A trained baseline cannot be replaced by random initialization.

Inference keeps the release normalization and applies the organizer's valid Gaussian smoothing. The kernel starts from a nominal length-100 impulse with sigma 2; coefficients strictly greater than 0.01 are retained and renormalized, leaving nine bins. Smoothing reduces the valid input by eight bins. Patches of 14 bins with stride four then produce `floor((T_raw - 22) / 4) + 1` frames. FP32 execution explicitly disables TF32. Recorded parity checks cover the smoothing, logits and valid prefixes of padded batches.

Native Flashlight performs lexicon-constrained CTC word search over a stress-stripped CMUdict lexicon. The Gilbreth baseline uses KenLM 0.3.0 with a three-gram estimated from official training transcripts. The Windows wheel could not load its KenLM DLL, so laptop word measurements use the explicitly named `lexicon_only`/ZeroLM configuration. Their WER and timing describe that fallback.

Flashlight 0.0.7 limits a pronunciation terminal to six word labels. With sorted insertion, the original trie retains 134,487 of 134,711 pronunciations and 125,716 of 125,924 dictionary words. The v2 external build removes that cap with dynamically sized terminal labels. The reference-free [trie audit](../scripts/audit_trie.py) reconstructs the reachable lexicon and verifies retention of every pronunciation and word. Pronunciation-lexicon OOV and training-LM OOV counts are reported separately.

The calibration study compares the unchanged checkpoint with head-only updates and with updates to the same head plus the existing session biases. The GRU backbone stays frozen, although input-bias gradients traverse it. Calibration requests cuDNN's training-mode backward buffers while explicitly setting recurrent dropout to zero. Inference runs with dropout off. Both v1 calibration conditions completed the same 720 updates from the official checkpoint; these are supervised adjustments within known sessions.

Candidate rescoring uses teacher-forced likelihoods of saved word hypotheses. It generates no new text. V1's confidence policies share one candidate cache. V2 compares pinned DistilGPT2 and Qwen models on the same selected lists, using Flashlight's acoustic component with a neural language prior. The [continuation report](../reports/continuation.md) records the scoring conventions, measured accuracy and complete-request timings.

The alternative encoders have separate model contracts and checkpoints:

- The completed 9.73-million-parameter Transformer was trained from scratch for 40,000 updates. It consumes the signed released inputs, uses causal attention over a bounded frame window, and exposes no streaming request interface. Its completed validation comparison includes a separately frozen addendum after a memory stop and resume.
- The frozen-GRU diphone adaptation uses a 40-by-40 joint head and a separate blank. Its initialization preserves the original monophone probabilities; training uses T15 transition and self-pair targets, then marginalizes to the official phone inventory. It and the matched monophone control each completed 1,024 updates on the same eligible training data. This is an adaptation of the published objective, not a reproduction of the full DCoND-LIFT system.

Neither alternative changes the baseline token map or its stored cache. Both completed with worse exploratory report accuracy than their respective GRU controls; the [aggregate evidence](../reports/continuation.json) retains those results and their uncertainty. The [continuation registration](CONTINUATION.md) records their data access and run contracts.

Use is limited to offline research on this one-participant release. There is no clinical or live-device integration and no claim of clinical reliability. Published test-set results belong to their external systems; v2's exploratory benchmark status follows from v1 report exposure.
