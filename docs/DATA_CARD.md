# Data card

BTT-25-CW uses the T15 neural-training release from [Data for: An Accurate and Rapidly Calibrating Speech Neuroprosthesis](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), Dryad version 374060, published 3 July 2025. The associated paper is [Card et al., NEJM 2024](https://doi.org/10.1056/NEJMoa2314132).

The recordings capture attempted speech from implanted cortical electrodes in one participant. Each 20 ms bin contains 512 threshold-crossing and spike-band-power features derived from 256 channels. The released features were normalized using the preceding 20 trials. These processed intracortical signals support attempted-speech decoding.

| Official partition | Trials | Session files | Labels in the release |
|---|---:|---:|---|
| Train | 8,072 | 45 | Text and phonemes |
| Validation | 1,426 | 41 | Text and phonemes |
| Test | 1,450 | 41 | Hidden |

All partitions use the checkpoint's fixed map of 45 known sessions. A session absent from validation or test retains its original day index. HDF5 access is lazy, and declared input and target lengths determine the valid prefixes of padded storage. Trial IDs preserve session, original partition and source group. Test export follows checkpoint session order and the captured HDF5 key traversal.

Before the new project's reference-based experiments, validation was divided by session/block into 926 `dev_tune` and 500 `dev_report` trials. The exact memberships stay in a local manifest with a portable hash. During v1 development, feature-only inference could cover all validation trials, while scores and examples were restricted to `dev_tune` until the joint study freeze. The evaluator checks the scoring selection before loading references.

Two exposures limit interpretation. The inherited GRU was selected on official validation and had seen training data from all known sessions. Then v1 reporting results informed the design of v2. The original local split protected v1's new choices; v2 scores on both `dev_report` and full validation are exploratory. These experiments do not establish unseen-day generalization or an evaluation independent of historical checkpoint selection. The [continuation registration](CONTINUATION.md) records that history.

The project-trained KenLM three-gram uses only the 8,072 official training transcripts. Local model fitting and weight selection use their declared train/tuning selections. DistilGPT2 and Qwen are general pretrained models with no local task fine-tuning; their incidental overlap with benchmark sentences is unknown. The LightBeam feasibility run used the same training-only three-gram with Qwen. A separate ImagineVille four-gram was acquired for v2, but neither sustained pilot completed admission or produced a tuning score. Its external training corpus is distinct from the project's permitted task text.

Raw recordings, reference transcripts, reference-bearing inventories and submission CSVs stay outside Git. Public reports contain aggregate metrics and selected model outputs without reference labels. Failed and empty predictions remain in evaluation denominators. Obtain the release from its authorized source, follow its terms, and retain the paper and dataset citations; see [PROVENANCE.md](PROVENANCE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
