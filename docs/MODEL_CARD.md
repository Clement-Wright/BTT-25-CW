# Model card

The baseline is the official pretrained T15 GRU, loaded strictly from a checksum-verified release. It has 44,315,177 parameters: 45 day-specific 512-dimensional affine transforms with Softsign, five unidirectional 768-unit GRU layers, a learned initial state, and a 41-class phoneme head. Blank is 0, AA is 1, and SIL/boundary is 40. Padding has no output class.

Inference preserves released normalization. The Gaussian kernel is constructed from a nominal length-100 impulse with sigma 2, thresholded at coefficients greater than 0.01 and renormalized; its effective support is nine bins. Valid smoothing reduces input length by eight, then 14-bin patches with stride four give `floor((T_raw - 22) / 4) + 1` output frames. FP32 execution disables TF32 explicitly to improve real padded-batch invariance.

The checkpoint stores a uniform `_orig_mod.` prefix from `torch.compile`. Only that documented wrapper is removed; loading remains strict. The original NumPy scalar metadata requires a narrow restricted-loading allowlist. Missing or incompatible weights raise errors. The trained model is never replaced with random initialization.

The compact word path uses native Flashlight lexicon CTC search with a stress-stripped CMUdict phone lexicon. Gilbreth used KenLM 0.3.0 with a three-gram language model estimated on official training transcripts. The Windows build lacked a KenLM DLL, so the laptop word benchmark used the explicitly labeled `lexicon_only` fallback. Candidate rescoring experiments retain exactly the same cached word candidates.

Flashlight 0.0.7 retains six word labels per pronunciation. With sorted insertion, its trie keeps 134,487 of 134,711 entries and 125,716 of 125,924 dictionary words. The audit in `scripts/audit_trie.py` reconstructs the reachable lexicon without references. The cluster's reporting runs use this inventory for lexicon OOV counts and track KenLM OOV separately.

Small calibration compares the unchanged checkpoint, head-only updates, and the same head plus existing session bias updates. Session bias updates are a small adaptation of the established day transform, rather than a new large network. The backbone remains frozen, but gradients must traverse it for input calibration. GRU training mode supplies cuDNN backward buffers; recurrent dropout is explicitly zero in these calibration conditions. Inference dropout remains off.

The [accuracy and efficiency continuation](CONTINUATION.md) preserves that official loader. Its alternative checkpoints use separate model contracts. The compact Transformer uses signed released inputs, causal attention over a bounded frame window, and no streaming interface. The frozen-GRU diphone adaptation uses the published joint-distribution idea with T15-specific transition and self-pair targets. Its monophone control uses matched eligible training data and update count. Neither contract changes the official token map or stored baseline cache.

This is offline research on one participant. It has no clinical/live-device integration or claim of clinical reliability. Published leaderboard values describe external test evaluations. V2 exploratory evaluation on the existing benchmark carries the v1 exposure disclosure.
