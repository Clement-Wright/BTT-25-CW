# Run on Gilbreth

The checked-in `hpc/` scripts use CPython 3.12 from `anaconda/2024.10-py312`, `uv/0.8.0`, and PyTorch 2.8.0+cu126. Baseline/scorer and native-decoder packages live in separate scratch environments. These instructions launch bounded stages; they do not claim a completed study or a cluster-wide performance result.

Use your existing Gilbreth access and normal host-key/MFA procedure. Discover your allowed account and partition with `slist`, `sinfo` and the current RCAC documentation. Set `ACCOUNT` and `PARTITION` in your shell from that discovery. Gilbreth requires an account and partition; `standby` is a QOS with a four-hour limit and access to idle resources. See [RCAC's account, partition and QOS rules](https://docs.rcac.purdue.edu/userguides/gilbreth/run_jobs/queues/).

Run setup, downloads, archive hashing, model work and tests inside Slurm allocations. The login shell only prepares source paths and submits or inspects jobs.

## Scratch layout and source

All scripts derive the project root from `$RCAC_SCRATCH`; don't substitute a backing filesystem path. Scratch is temporary and has no backup. Export verified results and restart checkpoints to durable storage before removing any scratch copy. See [RCAC scratch guidance](https://docs.rcac.purdue.edu/userguides/gilbreth/storage/scratch_space/).

```bash
umask 077
project="${RCAC_SCRATCH:?}/btt-25-cw"
mkdir -p "$project"/{data,receipts,slurm,cache/tmp,environments}
git clone "${REPOSITORY_URL:?Set the public source repository URL}" "$project/workspace"
cd "$project/workspace"
: "${ACCOUNT:?Set your allowed Slurm account}"
: "${PARTITION:?Set an allowed GPU partition}"
```

For an existing source checkout, use its `workspace` directory instead of cloning again. Transfer only the public source and required authorized assets; keep credentials, unrelated checkout files and machine inventories out of the transfer. Do not copy Windows virtual environments or reuse a Windows `configs/local.yaml` on Linux. The preparation stage generates cluster-local paths and the same frozen membership from the verified release.

The layout created by the scripts is:

```text
$RCAC_SCRATCH/btt-25-cw/
  workspace/             public source, ignored local assets and run outputs
  data/                  verified release archive and expanded HDF5 files
  environments/baseline/ Python 3.12, CUDA Torch and scorer stack
  environments/decoder/  Python 3.12, NumPy, psutil and Flashlight wheel
  environments/kenlm/    pinned estimator tools and shared libraries
  cache/                 uv, Hugging Face, temporary files and KenLM build
  receipts/              setup, acquisition and build receipts
  slurm/                 scheduler logs and stop requests
```

## Bounded setup and data preparation

The examples request one GPU, four CPUs and 16 GB of host RAM. Choose a partition you can access and recheck its limits before submitting. GPU VRAM and the requested host allocation are measured separately.

```bash
slurm=(--account="$ACCOUNT" --partition="$PARTITION" --qos=standby
       --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=4 --mem=16G
       --kill-on-invalid-dep=yes
       --output="$project/slurm/%x-%j.out")

setup_id=$(sbatch --parsable "${slurm[@]}" --time=00:45:00 --job-name=btt-setup hpc/gilbreth_setup.sh)
data_id=$(sbatch --parsable "${slurm[@]}" --dependency="afterok:$setup_id" --time=01:30:00 --job-name=btt-data hpc/gilbreth_data.sh)
prepare_id=$(sbatch --parsable "${slurm[@]}" --dependency="afterok:$data_id" --time=00:30:00 --job-name=btt-prepare hpc/gilbreth_run.sh prepare "$project/data/archive-expanded/hdf5_data_final")
kenlm_id=$(sbatch --parsable "${slurm[@]}" --dependency="afterok:$prepare_id" --time=00:30:00 --job-name=btt-kenlm hpc/gilbreth_kenlm.sh)
printf '%s\n' "$setup_id" "$data_id" "$prepare_id" "$kenlm_id"
```

`gilbreth_setup.sh` installs the baseline requirements, pinned scorer packages and CUDA Torch. It obtains the CPython 3.12 manylinux Flashlight Text 0.0.7 wheel through PyPI metadata and checks its download hash. It then acquires the official model, pinned CMUdict/scorer assets and lexicon. Its receipt records package versions and the allocated GPU check.

`gilbreth_data.sh` resumes a partial download of the pinned independent mirror, verifies the full archive SHA256, then extracts only the released HDF5 members against [assets/t15_expanded_sha256.json](../assets/t15_expanded_sha256.json). `prepare` re-verifies those files and runs `prepare_local.py`. Both operations avoid reference dataset reads. The data and model terms in [REPRODUCE.md](REPRODUCE.md) still apply.

`gilbreth_kenlm.sh` verifies the KenLM 0.3.0 source archive and builds the estimator and shared libraries under `environments/kenlm`. It uses Anaconda's Boost libraries and expects CMake on the allocated node. The script records tool/library hashes and dependency output. A successful build must provide `bin/lmplz`, `bin/build_binary` and loadable KenLM shared libraries; no system installation or administrator access is needed.

The dependency chain stops after a failed prerequisite. Inspect the failed job and its receipts before submitting a successor. Slurm dependency IDs schedule jobs; they do not certify scientific completion.

## Submit the measured stages

Use the wrapper for every scientific stage. It fixes the working directory and cache locations, enables offline scorer loading, sets CPU thread counts, and handles scheduler stop signals. The helper below only submits a finite command; it does not retry or resubmit itself.

```bash
decoder="$project/environments/decoder/bin/python"
kenlm_bin="$project/environments/kenlm/bin"

submit_stage() {
    local name="$1" wall="$2" previous="$3"
    shift 3
    sbatch --parsable "${slurm[@]}" --dependency="afterok:$previous" \
        --job-name="$name" --time="$wall" --signal=B:USR1@120 \
        hpc/gilbreth_run.sh python "$@"
}

baseline_id=$(submit_stage btt-baseline 00:20:00 "$kenlm_id" \
    -m btt25.cli infer --config configs/local.yaml --split val --run-dir runs/baseline-val)
head_pilot_id=$(submit_stage btt-pilot-head 00:25:00 "$baseline_id" \
    -m btt25.cli profile --config configs/local.yaml --minutes 15 --mode head --run-dir runs/profile-head)
bias_pilot_id=$(submit_stage btt-pilot-bias 00:25:00 "$head_pilot_id" \
    -m btt25.cli profile --config configs/local.yaml --minutes 15 --mode session_bias --run-dir runs/profile-bias)
compact_id=$(submit_stage btt-compact 01:00:00 "$bias_pilot_id" \
    scripts/run_studies.py compact --config configs/local.yaml --decoder-python "$decoder" \
    --kenlm-bin "$kenlm_bin" --logit-run runs/baseline-val --lexicon assets/local/lexicon.txt --run-dir runs/compact)
```

The compact command first checks the real native KenLM import and tiny decoder contracts. It fits a three-gram modified Kneser-Ney LM using normalized official-training transcripts only, with a 1 GB estimator memory request. It selects the compact weight on `dev_tune`; a requested backend failure stops the stage. Natural-log LM weights are converted to the native KenLM scale inside the worker.

To use the declared fallback instead, omit `--kenlm-bin` and add `--lexicon-only` in a separate run directory. Keep its label in all comparisons. Do not substitute it midway through a frozen KenLM condition.

Continue only after the successful pilots and complete compact selection exist:

```bash
confidence_id=$(submit_stage btt-confidence 01:00:00 "$compact_id" \
    scripts/run_studies.py confidence-tune --config configs/local.yaml --decoder-python "$decoder" \
    --selection runs/compact/selection.json --scorer-dir assets/local/distilgpt2 --run-dir runs/confidence-tune)
head_id=$(submit_stage btt-head 01:05:00 "$confidence_id" \
    scripts/run_studies.py calibration --config configs/local.yaml --decoder-python "$decoder" \
    --selection runs/compact/selection.json --mode head --profile-dir runs/profile-head --run-dir runs/calibration-head)
bias_id=$(submit_stage btt-bias 01:05:00 "$head_id" \
    scripts/run_studies.py calibration --config configs/local.yaml --decoder-python "$decoder" \
    --selection runs/compact/selection.json --mode session_bias --profile-dir runs/profile-bias --run-dir runs/calibration-bias)
test_id=$(submit_stage btt-test 00:30:00 "$bias_id" \
    scripts/run_studies.py test --config configs/local.yaml --decoder-python "$decoder" \
    --selection runs/compact/selection.json --run-dir runs/test)
```

Both calibration conditions retain the fixed settings in `configs/calibration.yaml`; admission uses their matching sustained pilot. Each pipeline uses one supervisor for training, full validation inference, native decoding and `dev_tune` evaluation. The wrapper currently requests a 60-minute target and a 62-minute internal watchdog, leaving scheduler time around it. A100 execution uses the same FP32/TF32 settings as the native baseline; cross-device equality is not guaranteed.

The test stage saves `runs/test/submission.csv` in official order from features alone. It does not calculate test WER.

For a lower-level pipeline, inspect `python -m btt25.pipeline --help` and use `--dry-run`. Always pass `--decoder-python "$decoder"`; the cluster's decoder lives beside the baseline environment rather than in the source checkout's `.decoder-venv`.

## Joint freeze and reporting

Seal after both calibration pipelines are complete and the confidence policy is fixed. The report driver verifies the matched update counts, calibration IDs, source checkpoint and live artifacts before allowing reporting labels. Use one declaration for all phases:

```bash
srun --account="$ACCOUNT" --partition="$PARTITION" --qos=standby \
    --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=2 --mem=4G --time=00:10:00 \
    "$decoder" scripts/audit_trie.py --lexicon assets/local/lexicon.txt --output-dir runs/trie-audit
report_args=(--config configs/local.yaml --selection runs/compact/selection.json
             --confidence runs/confidence-tune --head runs/calibration-head
             --bias runs/calibration-bias --scorer-dir assets/local/distilgpt2 --run-dir runs/locked
             --effective-lexicon runs/trie-audit/searchable_lexicon.txt)

seal_id=$(submit_stage btt-seal 00:20:00 "$test_id" \
    scripts/report_studies.py seal "${report_args[@]}")
deploy_id=$(submit_stage btt-report-a 01:00:00 "$seal_id" \
    scripts/report_studies.py deploy "${report_args[@]}")
summary_id=$(submit_stage btt-summary 00:30:00 "$deploy_id" \
    scripts/report_studies.py summarize "${report_args[@]}")
```

These commands open `dev_report` only after the joint freeze. Keep the resulting registry, freeze, configurations, resource summaries, comparisons and failed-condition evidence. See [REPRODUCE.md](REPRODUCE.md) for the reporting boundary and the interpretation of the inherited checkpoint.

## Monitor, stop and resume

Inspect recorded job IDs with lightweight scheduler commands:

```bash
squeue -j "$baseline_id,$head_pilot_id,$bias_pilot_id,$compact_id"
sacct -j "$summary_id" --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Read the corresponding `slurm/` log and the stage's `run_manifest.json` or `pipeline_manifest.json`. An allocation ending successfully does not turn `diagnostic_partial` or `budget_stopped` into a completed condition. Preserve preemption, timeout and OOM receipts.

The scheduler warning reaches `gilbreth_run.sh`, which writes its job-specific `BTT_STOP_FILE`. A nested pipeline supervisor mirrors an outer stop request to the child's stop file. Training saves at a completed update, and decoding stops between trials. Abrupt preemption can bypass that warning; atomic checkpoints and validated caches remain necessary. A native decoder call cannot stop midway through a trial.

Resume a calibration condition by submitting its identical `run_studies.py calibration` command as a new, explicitly tracked job. Reuse its existing run directory, selection, pilot and source configuration. The pipeline skips verified complete stages and resumes training with full saved state; the new attempt retains the preceding manifests and resource receipts. Do not run two writers against one condition directory.

For interrupted inference, resubmit the same `infer` command and run directory. For partial compact tuning before selection, repeat the compact command with the same assets; saved decoder trials undergo their cache checks. Once `selection.json` or a joint freeze exists, preserve it. Run a new sustained pilot in a new directory if the old pilot did not qualify.

The finite dependency examples use `afterok` to stop on error; resuming an interrupted prerequisite requires an explicit new submission and a deliberate successor chain. Requeue alone does not restore optimizer state. Cancel only an exact job you own, after requesting a safe stop and checking the latest checkpoint. Never delete run evidence to make a retry appear fresh.

Export the verified source revision, frozen metadata, manifests, metrics, logs and required restart/final checkpoints to authorized durable storage. Keep downloaded neural data and reference-bearing caches private under their release terms. Scratch alone is not a completed archive.
