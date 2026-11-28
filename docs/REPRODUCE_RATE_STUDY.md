# Reproduce the rate study on Gilbreth

Start with the verified T15 data manifest, local split manifest, official checkpoint, patched Flashlight binding, lexicon and train-only KenLM produced by the [Gilbreth workflow](GILBRETH.md). The stronger decoder also uses the v2 freeze and the pinned DistilGPT2 assets. All model inference in these Slurm examples runs on the compute node.

The scripts use `$RCAC_SCRATCH/btt-25-cw/workspace` for the established assets, `environments/baseline` for PyTorch, `environments/decoder` for Flashlight and `v4-rate-20261127` for this study. Stage the Python package under `src/btt25`, the native server and worker under `scripts`, and the v4 job scripts under `hpc` in the study directory. Keep each attempt in its own output directory.

```bash
export RCAC_SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/$USER}"
cd "$RCAC_SCRATCH/btt-25-cw/v4-rate-20261127"
mkdir -p logs
job=(sbatch --parsable -A ieee-smcs -p a100-40gb -C nvlink --qos=normal
     --gres=gpu:1 --cpus-per-task=4 --mem=15000M --time=01:40:00 --signal=B:USR1@100)
"${job[@]}" --job-name=btt4-midpoint -o 'logs/midpoint-%j.log' \
  hpc/gilbreth_v4_rate.sh two_rate --conditions two_rate \
  --updates 2048 --pilot-seconds 900 --max-seconds 5400
```

Use `gru80` or `gru40` in both condition positions for the matched controls. The array launcher lists the original twelve conditions. For the separately registered 20 ms control, submit `"${job[@]}" hpc/gilbreth_v4_dense.sh` from the same directory. That entry point registers the one-bin stride before it invokes the trainer. The checkpoint stores the trainable weights and AdamW state. Attach a verified parent with the same training and feature contracts when using a delta for inference. To continue an interrupted run, append `--resume-from "$PWD/runs/<old-attempt>/study"` to the first command. The new job writes a new attempt and checks the source digest and training membership before loading it.

Each completed arm writes `study/<condition>/metrics.json`, `predictions.jsonl`, `logits/`, `resources.json` and `checkpoints/last.pt`. The per-trial predictions contain private text. Keep the data, checkpoints, optimizer state and candidate lists on authorized storage.

The original Gilbreth training snapshot used CRLF line endings in two Python files. Git stores the same source with LF. New runs can use LF throughout. To resume an archived campaign checkpoint in a new scratch copy of its source tree, restore those two byte identities before invoking the resume command. The following compares the resulting bytes with the submitted hashes before writing either file.

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path

expected = {
    "v4_rate.py": "75f4632a762e88de9ff07d2675f05afcf4e7902eeb6f60b58182d06ecb03b319",
    "v4_rate_study.py": "c4af69849b6432bd50dfcfac4375eefbdf1ed51126b85f83bc5df34edf80b55e",
}
copies = {}
for name, digest in expected.items():
    path = Path("src/btt25") / name
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Different source in {path}")
    copies[path] = data
for path, data in copies.items():
    path.write_bytes(data)
PY
```

The frozen clock diagnostic uses `hpc/gilbreth_v4_rate.sh clock`. The comparison launcher `hpc/gilbreth_v4_compare.sh` requires one completed 2,048-update checkpoint per condition and the original smoothing and clock-control runs identified in the script. It starts a new model process per pass and reuses that process and its decoder across the warm requests. Timing starts before the HDF5 read and ends after the final candidate selection and device synchronization. Each pass verifies its text against the saved scoring run. The summary retains replay differences.

The cadence launchers use named source snapshots. On a fresh installation, place `src/btt25/v4_rate_decode.py` and `src/btt25/v4_rate_analysis.py` under `cadence-code-v1/` in the study directory. Place `src/btt25/v4_rate_replay_audit.py` and its sibling `v4_rate_decode.py` under `cadence-score-v1/`. For the original training array, these commands reproduce blind search and the score-only audit with your new job IDs.

```bash
blind_job=$(sbatch --parsable hpc/gilbreth_v4_rate_decode.sh "${train_array_id}_0" "${train_array_id}_2")
# If the lexical replay assertion stopped scoring after all 11,408 blind predictions, run the audit.
sbatch hpc/gilbreth_v4_rate_score.sh "$blind_job" "$gru40_checkpoint_job" "$two_rate_checkpoint_job"
```

The two checkpoint job IDs are the numerical suffixes of the completed `runs/gru40-*` and `runs/two_rate-*` directories. The audit checks input hashes and transcript coverage before reporting the lexical replay differences. For the stronger decoder, use the [feature-to-text inference command](../reports/phone_robustness.md#reproduce-on-the-established-gilbreth-workspace). It takes a released trial manifest and the published model contracts. Rebuilding the historical timing comparisons additionally requires the private intermediate runs named in each launcher.

To regenerate the main figures from the public aggregates:

```bash
python scripts/render_rate_v4.py --summary reports/rate_v4.json
```
