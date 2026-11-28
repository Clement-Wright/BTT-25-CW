# Reproduce the laptop CPU measurements

Start in the project checkout with the authorized T15 features, original checkpoint, frozen split, CMUdict lexicon and local configuration from the [setup guide](REPRODUCE.md). The encoder uses four Torch CPU threads. Both precision settings use the same recorded bins, day transform, nine-bin valid smoothing, 14-bin patch and stride of four. The INT8 conversion packs the weights of the five GRU layers and their output head. Day transforms, the hidden state and emitted phone scores remain floating point.

## Native Windows

This route uses the official Flashlight Text 0.0.7 Windows wheel with the lexicon-only search. Run each new campaign in an unused directory.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.venv/Scripts/python -m btt25.consumer_v4 --mode pilot --out runs/cpu-zero-pilot
.venv/Scripts/python scripts/run_consumer_v4.py --pilot runs/cpu-zero-pilot --campaign runs/cpu-zero
```

The pilot alternates both precision settings on 12 real tuning utterances sampled across the length range, saves both converted state files and decodes for at least 900 seconds. The driver checks the resource receipt and estimates time for scoring, three timing passes and two sensor-clock replays. A failed attempt keeps its directory. After inspecting its receipt, rerun the driver with `--retry-failed` to place another attempt in a separate directory. Before reusing an earlier pilot or completed condition, the driver checks the CPU model, runtime versions, split and asset identities. The first Windows pilot predates some of those fields. Its finished campaign can be verified again from the saved condition receipts.

## Repaired search and the training three-gram under WSL

Use a compatible Linux x86-64 CPython 3.12 runtime with PyTorch 2.8, NumPy, SciPy, h5py, PyYAML and psutil. Start from the repaired Flashlight and KenLM build in the [Gilbreth setup](GILBRETH.md). The staging command below expects the project’s archived repaired build (search job 12523602) and its train-only three-gram. For a different source location, set `BTT_REPAIRED_PACKAGES` and `BTT_TRAIN_TRIGRAM` to the corresponding absolute cluster paths. Supply the cluster project directory and the Boost library directory from that build's dependency listing. Keep these assets private.

```bash
# Run in WSL, from the project checkout on the Windows drive.
bash hpc/stage_consumer_decoder.sh "$control_socket" "$cluster_login" \
  "$cluster_project" "$cluster_boost_lib" assets/local/v4-wsl-decoder

# Only if this environment lacks h5py, install it under the project.
uv pip install --python "$cpu_python" --target assets/local/v4-wsl-decoder/hdf5 \
  --no-deps h5py==3.15.1

bash scripts/run_consumer_wsl.sh "$cpu_python" assets/local/v4-wsl-decoder runs/cpu-kenlm
```

The WSL script verifies the transferred files against their checksums and the three-gram against the binary used for the published measurements. It checks that seven homophones can occupy one trie terminal and translates Windows HDF5 paths to their mounted locations in a private copy of the manifest. It keeps the original trial IDs and verifies every translated file exists. The first scoring run strictly loads the checkpoint. The decoder uses a beam of 50, a natural-log LM weight of 2 and the published phone and blank map.

For the Windows-side measurement of the WSL virtual machine, open a second PowerShell terminal in the checkout after the pilot directory contains `config.resolved.yaml`. Run the following alongside the campaign. It samples the working set of the entire VM every two seconds until the campaign ends.

```powershell
.\scripts\measure_wsl_vm.ps1 -Campaign runs/cpu-kenlm -Output runs/cpu-kenlm-pilot/windows-vm.jsonl
```

If you already have native Linux HDF5 paths and a local repaired Flashlight build, use those directly. Build the three-gram once from the authorized training split, or reuse its verified binary. Use a CPython 3.12 environment with the native extension's ABI. The loader must resolve the KenLM and Boost shared libraries for that build.

```bash
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src:$repaired_build/site-packages"
export LD_LIBRARY_PATH="$kenlm_prefix/lib:$repaired_build/site-packages/flashlight/lib/text:${LD_LIBRARY_PATH:-}"
"$cpu_python" scripts/fit_kenlm.py --data-manifest "$linux_manifest" \
  --kenlm-bin "$kenlm_prefix/bin" --output-dir runs/cpu-trigram
args=(--config "$linux_config" --decoder-python "$cpu_python" \
      --kenlm-model runs/cpu-trigram/train-3gram.binary)
"$cpu_python" -m btt25.consumer_v4 --mode pilot --out runs/cpu-kenlm-pilot "${args[@]}"
"$cpu_python" scripts/run_consumer_v4.py --pilot runs/cpu-kenlm-pilot \
  --campaign runs/cpu-kenlm "${args[@]}"
```

```bash
python scripts/report_consumer_v4.py --pilot runs/cpu-kenlm-pilot --campaign runs/cpu-kenlm \
  --windows-pilot runs/cpu-zero-pilot --windows-campaign runs/cpu-zero
```

To redraw the two figures from the checked-in aggregate alone, run `python scripts/report_consumer_v4.py --aggregate reports/consumer_v4.json`.

Every score run writes one final phone sequence and word prediction per tuning trial before opening the tuning references. The timing run alternates FP32 and INT8 after warmup. Isolated scoring processes provide each condition's memory measurements. The replay opens each recording's feature dataset and reads one row per 20 ms interval. The supplied end of a recording finalizes the utterance. Draft transcripts and final predictions stay in the private run directories.

To run the native Windows INT8 replay alone:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.venv/Scripts/python -m btt25.consumer_replay --precision int8 --out runs/cpu-int8-replay-only
```

`OnlineGRU.push(bins)` accepts a nonempty FP32 tensor of newly arrived T15 rows. Construct one instance per trial with its known session index. Before 22 rows have arrived it returns `None`. It then emits one 41-class vector every four rows. The native word search has matching `begin`, `step` and `end` messages. A partial word sequence can change at the next update. The replay records each change and compares the final incremental result with batch search on the identical accumulated emissions.

## Replay the trained two-rate model

Finish the 2,048-update two-rate condition in the [rate study](REPRODUCE_RATE_STUDY.md). In the cluster allocation, run the following command with its completed study directory, for example `runs/two_rate-<job>/study/two_rate`.

```bash
python scripts/export_rate_delta.py "$completed_two_rate" "$export_dir"
```

Copy only `two_rate.pt` and `delta.json` from this directory into `assets/local/v4-two-rate-delta`. The script verifies the complete training checkpoint before writing the trainable tensors. In WSL, replay the same eight recordings with the original 80 ms GRU update and the learned intermediate emission:

```bash
bash scripts/run_consumer_wsl.sh "$cpu_python" assets/local/v4-wsl-decoder \
  runs/cpu-kenlm assets/local/v4-two-rate-delta
python scripts/report_consumer_v4.py --pilot runs/cpu-kenlm-pilot --campaign runs/cpu-kenlm \
  --windows-pilot runs/cpu-zero-pilot --windows-campaign runs/cpu-zero \
  --two-rate-replay runs/cpu-kenlm-two-rate-replay
```

At each 40 ms midpoint, the online adapter subtracts the mean of the previous two transformed bins from the mean of the latest two. It combines this change with the last GRU output and adds the learned residual to the previous anchor logits. The following anchor advances the recurrent state. The replay checks the delta and parent checkpoint hashes, exact parameter names, tensor shapes, training endpoint and model contract.
