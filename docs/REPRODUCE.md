# Reproduce the archived v1 pipeline

Use the separate v2 guide for the search repair, LightBeam, and feature-to-text latency experiments. [V2 runbook](CONTINUATION.md) · [V2 results](../reports/continuation.md)

Run the commands from the repository root. Keep downloaded assets, references, resolved machine paths, and run artifacts out of tracked files. Check the stage manifests for run status.

The official checkpoint was developed using the known sessions and official validation. This project reserved `dev_report` references until the v1 freeze. The partition cannot show unseen-day generalization. [Data card](DATA_CARD.md) · [Model card](MODEL_CARD.md)

## Native Windows environments

Use 64-bit Windows, `uv`, and a CUDA-capable NVIDIA driver for GPU execution. The baseline dependency snapshot comes from CPython 3.13.5 with PyTorch 2.8.0+cu126. Flashlight uses a separate CPython 3.12 environment because its pinned Windows wheel does not support Python 3.13.

Create the baseline environment in PowerShell.

```powershell
New-Item -ItemType Directory -Force assets/local | Out-Null
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'assets/local/uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) 'assets/local/python'
uv venv --no-project --python 3.13.5 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0+cu126'
uv pip install --python .venv/Scripts/python.exe -r configs/requirements-baseline.txt
$Py = (Resolve-Path .venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $Py -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
```

Set up the decoder next. The script finds a suitable interpreter or installs one under the project, installs wheels, and checks native imports.

```powershell
& ./scripts/setup_decoder.ps1
$Decoder = (Resolve-Path .decoder-venv/Scripts/python.exe).Path
& $Decoder -m unittest discover -s tests -p test_decoder_worker.py -v
```

The core Windows Flashlight route supports lexicon search through `ZeroLM`. A missing KenLM DLL leaves this route available. An explicitly requested KenLM model must load successfully; the worker does not replace a failed KenLM request with another decoder. Use [GILBRETH.md](GILBRETH.md) for the Linux KenLM build.

## Obtain and verify the T15 release

Obtain access under the [Dryad release's terms](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). The source repository includes acquisition metadata and checksums, not the neural data or pretrained weights. [assets/manifest.json](../assets/manifest.json) records the canonical source and the pinned independent mirror.

Download and verify the model.

```powershell
& $Py -m btt25.cli acquire-model
```

Place your authorized Dryad archive at the path below or fetch the pinned mirror. The archive hash must match before extraction.

```powershell
$Archive = 'assets/local/t15_copyTask_neuralData.zip'
$DataUrl = 'https://huggingface.co/datasets/yw784/brain-to-text-25-competition/resolve/2f2fabacbc9af637da3bac483bf14e1231fb4e8a/t15_copyTask_neuralData.zip'
$Expected = (Get-Content assets/manifest.json -Raw | ConvertFrom-Json).official_data.sha256
if (-not (Test-Path -LiteralPath $Archive)) {
    curl.exe --fail --location --retry 2 --continue-at - --output "$Archive.partial" $DataUrl
    if ($LASTEXITCODE -ne 0) { throw 'Data download failed' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath "$Archive.partial").Hash.ToLowerInvariant() -ne $Expected) {
        throw 'Data archive checksum mismatch'
    }
    Move-Item -LiteralPath "$Archive.partial" -Destination $Archive
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant() -ne $Expected) {
    throw 'Existing data archive checksum mismatch'
}
& $Py scripts/expand_release.py --archive $Archive --data-root assets/local/data/hdf5_data_final --hashes assets/t15_expanded_sha256.json --receipt assets/local/release_receipt.json
& $Py scripts/verify_release.py --data-root assets/local/data/hdf5_data_final --hashes assets/t15_expanded_sha256.json --output assets/local/release_verification.json
```

Extraction and verification compare released filenames, sizes and file hashes without opening reference datasets. Existing mismatching files produce errors. If you already have an expanded authorized release, point `--data-root` at its `hdf5_data_final` directory and run verification there.

Acquire the pinned dictionary and corpus metadata first so preparation can include the metadata path in the generated config.

```powershell
& $Py scripts/acquire_extras.py
& $Py scripts/prepare_local.py --data-root assets/local/data/hdf5_data_final --device cuda
& $Decoder scripts/decode_worker.py --build-lexicon-only --cmudict assets/local/cmudict/cmudict.dict --lexicon assets/local/lexicon.txt
```

Preparation checks the model ZIP, extracted checkpoint and argument hashes. It writes ignored `configs/local.yaml`, a feature metadata manifest and the frozen validation membership. A changed existing split is an error. Keep the same split for all conditions; preparation does not load transcript or phoneme references.

## Baseline inference and explicit word fallback

Run feature-only inference to save FP32 logits in validation trial order. Then score phonemes on the tuning partition.

```powershell
& $Py -m btt25.cli infer --config configs/local.yaml --split val --run-dir runs/baseline-val
& $Py -m btt25.cli evaluate --config configs/local.yaml --predictions runs/baseline-val/predictions.jsonl --output-dir runs/baseline-tune-phones --score-split dev_tune --phoneme-only
```

Use `--limit` for a partial diagnostic and omit it for full coverage. The phoneme-only command reports PER.

On Windows, request the lexicon-only word baseline.

```powershell
& $Py scripts/run_studies.py compact --config configs/local.yaml --decoder-python $Decoder --logit-run runs/baseline-val --lexicon assets/local/lexicon.txt --lexicon-only --run-dir runs/compact
```

This saves the selected complete word candidate cache, tuning metrics and `runs/compact/selection.json`. Native search uses beam 50, all 41 token columns, blank 0, SIL 40, threshold 25, log-add and up to 10 unique word sequences. It forbids `<unk>` output. This explicit fallback has no trained n-gram LM.

On Linux, replace `--lexicon-only` with `--kenlm-bin` pointing at the installed estimator directory. The compact stage fits a three-gram model on normalized official-training transcripts only, then selects among the weights in [compact_decoder.yaml](../configs/compact_decoder.yaml) using summed `dev_tune` word edits. It preserves every tested weight. See [GILBRETH.md](GILBRETH.md) for the complete command.

## Run the studies before reporting

Install the scorer in the baseline environment and fetch its pinned assets before starting a timed run.

```powershell
uv pip install --python $Py --no-deps -r configs/requirements-lm.txt
& $Py scripts/acquire_extras.py --with-scorer
```

The confidence study rescores existing word candidates with teacher-forced DistilGPT2 likelihood. Its grouped fitting and selection subsets both lie inside `dev_tune`. Lambda zero measures A0. The grouped fitting slice chooses a positive weight for the three rescoring policies, and the selection slice checks the gate rule. The configuration fixes call fraction and length-matched random controls.

```powershell
& $Py scripts/run_studies.py confidence-tune --config configs/local.yaml --decoder-python $Decoder --selection runs/compact/selection.json --scorer-dir assets/local/distilgpt2 --run-dir runs/confidence-tune
```

Run a sustained pilot for each calibration mode on the same device, checkpoint, and input manifests you will use for training.

```powershell
& $Py -m btt25.cli profile --config configs/local.yaml --minutes 15 --mode head --run-dir runs/profile-head
& $Py -m btt25.cli profile --config configs/local.yaml --minutes 15 --mode session_bias --run-dir runs/profile-bias
```

Admission requires at least 15 sustained minutes, host memory headroom, 20% device-memory headroom and a forecast with a 30% time reserve. A stopped or failed pilot does not qualify. The two calibration conditions use the same allowed official-training IDs and fixed update count; selection uses the final checkpoint.

```powershell
& $Py scripts/run_studies.py calibration --config configs/local.yaml --decoder-python $Decoder --selection runs/compact/selection.json --mode head --profile-dir runs/profile-head --run-dir runs/calibration-head
& $Py scripts/run_studies.py calibration --config configs/local.yaml --decoder-python $Decoder --selection runs/compact/selection.json --mode session_bias --profile-dir runs/profile-bias --run-dir runs/calibration-bias
```

The calibration wrapper reads the shared settings and supervises training, full validation inference, fixed compact decoding, and `dev_tune` evaluation in order. It sets a 3,600-second target and a 3,720-second watchdog, and rejects forecasts over budget. Verify completion in `pipeline_manifest.json`, the stage manifests, and the resource receipts. [Calibration settings](../configs/calibration.yaml)

## Freeze, then open reporting labels

Finish compact selection, confidence tuning, and both calibration pipelines before sealing. Reuse the same arguments through reporting.

```powershell
& $Decoder scripts/audit_trie.py --lexicon assets/local/lexicon.txt --output-dir runs/trie-audit
$ReportArgs = @(
    '--config', 'configs/local.yaml',
    '--selection', 'runs/compact/selection.json',
    '--confidence', 'runs/confidence-tune',
    '--head', 'runs/calibration-head',
    '--bias', 'runs/calibration-bias',
    '--scorer-dir', 'assets/local/distilgpt2',
    '--effective-lexicon', 'runs/trie-audit/searchable_lexicon.txt',
    '--run-dir', 'runs/locked'
)
& $Py scripts/report_studies.py seal @ReportArgs
& $Py scripts/report_studies.py deploy @ReportArgs
& $Py scripts/report_studies.py summarize @ReportArgs
```

`seal` verifies completed calibration conditions and their matched training IDs/updates. It records example IDs, a live artifact registry and a joint freeze binding both studies before reporting references open. `deploy` runs the frozen confidence policies on `dev_report`; `summarize` produces the reporting comparisons and diagnostics for the declared conditions. Preserve negative and failed comparisons. A pre-existing joint freeze cannot be overwritten.

Manual `evaluate --score-split dev_report` or `official_val` also requires `--freeze-manifest` and `--artifact-registry`; the evaluator verifies actual files against the joint freeze before loading the selected references. Never replace those files with handwritten hash claims. Official-validation scores retain the benchmark-development limitations in the data card.

## Export hidden-test predictions and figures

Decode the released test features using the fixed compact selection.

```powershell
& $Py scripts/run_studies.py test --config configs/local.yaml --decoder-python $Decoder --selection runs/compact/selection.json --run-dir runs/test
```

`runs/test/submission.csv` contains `id,text` in official order. Test references are unavailable, so only order and coverage can be verified. A run using the Windows lexicon-only selection yields a lexicon-only submission.

Regenerate aggregate results and figures from the completed study directory. Include the verified submission to publish its coverage and hash. The submission itself stays in the run directory.

```powershell
uv pip install --python $Py 'matplotlib==3.10.6'
& $Py scripts/make_public_evidence.py --study-dir runs/locked --output-dir runs/public-evidence --submission A0=runs/test/submission.csv --expected-test-count 1450
```

## Redraw the figures

The checked-in aggregate results are enough to redraw all eight figures. With Matplotlib 3.10.6 installed, run this from the repository root.

```bash
python scripts/render_public_figures.py --reports reports
```

## Resume and inspect failures

Repeat an interrupted inference command with the same config and output directory. Repeat the decoder command stored in its run or pipeline manifest for interrupted word decoding. Their cache checks reject changed checkpoints, inputs, source hashes, vocabulary assets or trial identities. Preserve a rejected cache and use a new run directory when changing a condition; do not delete evidence to force a match.

For calibration, repeat the same `run_studies.py calibration` command. The pipeline verifies completed stages, resumes interrupted training through the existing resume API, and keeps attempt logs and prior resource snapshots. To inspect the planned commands without launching them:

```powershell
& $Py -m btt25.pipeline --config runs/calibration-head/calibration.resolved.yaml --decoder-python $Decoder --lexicon assets/local/lexicon.txt --mode head --profile-dir runs/profile-head --updates 720 --run-dir runs/calibration-head --target-seconds 3600 --hard-seconds 3720 --dry-run
```

To resume training outside the pipeline, use the same resolved calibration config and pilot.

```powershell
& $Py -m btt25.cli resume --config runs/calibration-head/calibration.resolved.yaml --run-dir runs/calibration-head/train --mode head --updates 720 --profile-dir runs/profile-head --max-hours 1
```

Use a new directory for a failed sustained pilot. Compact tuning refuses another selection after `selection.json` exists, and reporting sealing refuses another joint freeze. Stage manifests distinguish `complete`, `diagnostic_partial`, `budget_stopped` and `failed`. A word decoder failure remains in the word denominator while its separately valid greedy phonemes can still contribute to PER.

The pipeline checks a cooperative `BTT_STOP_FILE` between stages; training and decoding check at safe work boundaries. Host resources use a 12 GB operating target, a 16 GB ceiling and at least 2 GB available to the OS. Native decoding cannot interrupt a C++ call mid-trial, so retain an external watchdog. [GILBRETH.md](GILBRETH.md) explains the scheduler boundary.

After setup, run the source tests and the decoder tests in their respective environments.

```powershell
& $Py -m pytest
& $Decoder -m unittest discover -s tests -p test_decoder_worker.py -v
```

These checks do not replace the sustained pilots, complete candidate caches or frozen reporting artifacts.
