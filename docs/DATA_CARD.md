# Data card

Source: [Data for An Accurate and Rapidly Calibrating Speech Neuroprosthesis](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), T15 July 2025 release. Cite [Card et al., NEJM 2024](https://doi.org/10.1056/NEJMoa2314132).

The data records attempted speech from implanted cortical electrodes in one participant. It is not EEG, synchronized speech audio, or a general thought-reading dataset. There are 512 released threshold-crossing/spike-band-power features per 20 ms bin, derived from 256 channels. Released normalization uses the preceding 20 trials.

The recovered local release has 8,072 train, 1,426 validation and 1,450 hidden-label test trials across 45 known sessions. Test labels are unavailable. HDF5 is read lazily; true input/target lengths override padded storage. IDs retain session, original split and source group. Export follows checkpoint session order and exact HDF5 key traversal.

Before new reference-based work, validation was partitioned by session/block into 926 `dev_tune` and 500 `dev_report` trials. Exact membership is saved locally, with a portable split hash. Before the joint primary-study freeze, feature-only inference may cover all validation trials, but scores and examples may use only `dev_tune`. Reporting selection is checked before references are loaded.

The pretrained GRU saw all known sessions and was selected on official validation. This local report split protects new project choices only. It does not establish unseen-day generalization or independence from historical checkpoint selection. Full official-validation scores are benchmark-development results.

Task language priors use training transcripts only. Reporting text is excluded from task LM/gate fitting. DistilGPT2's general web pretraining has unresolved sentence overlap. Any LightBeam LM adapted on train plus validation text would require a different evaluation claim; that adapted LM is not part of this baseline.

Raw data and reference-bearing machine inventories are excluded from Git. Acquisition uses authorized public sources and their terms. Empty and failed predictions remain in evaluation denominators.
