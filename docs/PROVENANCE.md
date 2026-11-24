# Provenance

This project recovers an unfinished T15 research pipeline. The new repository has a fresh history. The original private/team repositories, local branches, checkpoints and uncommitted work remain intact.

| Component | Source | Treatment |
|---|---|---|
| Released neural features, transcripts and pretrained GRU | [Dryad T15 release](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), version 374060, July 2025 | Existing expanded data reused; weights acquired separately. Never bundled in Git. Exact acquisition hashes are in `assets/manifest.json`. |
| GRU architecture, checkpoint keys, inference smoothing, token/day contracts and official text normalization | [Neuroprosthetics-Lab/nejm-brain-to-text](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/980a5b7e96392210edf976d76bdb5d15d2e0e62b) | Separately written implementation of the documented method. Upstream source is acquired privately for numerical parity; no unlicensed upstream source file is vendored. |
| Native lexicon CTC search | [Flashlight Text](https://github.com/flashlight/text), Python package 0.0.7 | External installed dependency; its own license applies. No copied implementation. |
| Pronunciations | [CMUdict](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690) | External lexicon; stress digits removed, variants preserved, every phone checked. CMUdict's notice accompanies local acquisition. |
| Additional candidate scorer | [DistilGPT2](https://huggingface.co/distilbert/distilgpt2), revision `2290a62682d06624634c1f46a6ad5be0f47f38aa` | External Apache-2.0 weights, teacher-forced likelihood only. OpenWebText pretraining; unknown overlap with benchmark sentences. No task fine-tuning. |
| Historical experimental work | Clement-Wright/BTT_private_work, Purdue-IEEE-SMC/Brain-to-Text-2025, Clement-Wright/speechBCI_2024 | Audited and preserved; no old source or example result copied into the new baseline. Local checkpoint architectures/provenance do not support strict official-model reuse. |

The Purdue source was available as the current local clone, despite the connected remote API returning 404. The local branch differed from the handoff's inspected private default. Contributors in its history include Narendra Glasgow, CleWri00, Daniel Ng and corntoasted. The 2024-derived source credits TeamCyber/CIBR and Stanford organizers. These identities are historical credit, not a claim that they authored the new source tree.

No root redistribution license was found for the inspected legacy/official model source. Those files are excluded from this repository; a component's license does not cover neighboring material. Data, model and dictionary terms remain separate from this project's original code license.

The official ZIP's public Dryad stream returned 403 in this environment. An independent pinned Hugging Face mirror was used only after its complete archive hash matched authoritative Dryad metadata. This is checksum-verified recovery, not a claim that the mirror is operated by the dataset authors.

The released features are already normalized using preceding trials. This project adds only the checkpoint's required valid Gaussian smoothing. It applies no raw-signal high-gamma transform or per-utterance re-normalization.
