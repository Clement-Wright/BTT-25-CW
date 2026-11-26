# Reproduce the v2 comparisons

Keep the v2 checkout separate from `workspace`. The original validation logits, decoder assets, private artifact registry, and v1 freeze stay in `workspace`. [V1 release report](https://github.com/Clement-Wright/BTT-25-CW/blob/fe5fb4081f2ce256b41ba2a91f104a6a64a53deb/reports/results.md) · [V1 development results](https://github.com/Clement-Wright/BTT-25-CW/blob/fe5fb4081f2ce256b41ba2a91f104a6a64a53deb/reports/development.json)

The first experiment uses the same GRU emissions, training-only three-gram, and lexicon throughout. At each of three beam widths it compares the original Flashlight trie with the terminal-label repair. Oracles use the distinct candidates actually returned, up to the requested cutoff. An empty or failed hypothesis still contributes its reference words to the corpus denominator. [Search registration](../configs/search_v2.json)

Use the original `dev_tune` allowlist for diagnostics and seal the v2 artifacts before scoring the reporting partition. V1 already used the reporting benchmark, and the archived checkpoint had prior official-validation exposure. V2 reporting scores are exploratory.

Run language-model inference inside a Gilbreth allocation, with one heavy job at a time. Save the requested resources, effective process and cgroup limits, and scheduler receipts for every attempt alongside the warm timings. Keep caches, inventories, and trial-level evidence out of Git.

## Decoder and encoder comparisons

- The [search registration](../configs/search_v2.json) also defines one stronger first pass. It uses the standalone ImagineVille Dec 2019 Small four-gram with the same lexicon and beam-100 Flashlight settings. The [expanded search registration](../configs/search_expanded_v2.json) tests the four-gram against the three-gram at repaired beam 250. Incidental text overlap is unknown. [Publisher’s release and license](https://imagineville.org/software/lm/dec19/)
- The [fixed-candidate registration](../configs/fixed_candidates_v2.json) compares DistilGPT2 and Qwen3-1.7B-Base on identical saved candidates. KenLM steers candidate acquisition. The final score uses Flashlight’s acoustic component and the teacher-forced neural likelihood. Qwen uses exact candidate text with one BOS and an EOS target, no prompt or chat formatting, and FP32 accumulation over BF16 weights. The pinned DistilGPT2 control retains its v1 leading-space convention. Both models have unknown benchmark overlap from general pretraining. [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1)
- The [Qwen4B registration](../configs/fixed_candidates_4b_v2.json) requires the original scored packet hash, candidate limit, weight grid and batch size. Acquisition verifies all three shards, index and tokenizer files. [Qwen4B model card](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4b4dc7f14ee4320094d8b41684abff8539)
- The [LightBeam registration](../configs/lightbeam_v2.json) compares the same pinned backend in three modes on a fixed tuning subset. The periodic condition refreshes the neural prior every ten emission frames and applies final scoring with the same scorer as final-only scoring. The checkout does not supply the paper’s adapted model weights. Upstream code stays separate from this source tree. Paper results on the T15 public and private test segments used a different machine and model setup. [LightBeam paper](https://arxiv.org/html/2603.14002v1) · [BrainAudio source revision](https://github.com/ebrahimfeghhi/brainaudio/tree/90e0ac086d4dee643be154f446a6c374600575d7)
- The [encoder registration](../configs/encoders_v2.json) compares an offline causal Transformer trained from scratch with the official GRU under the compact decoder. It uses the published 384-wide, five-layer, six-head starting point and the T15 80 ms patch rate. The implementation operates directly on the signed standardized feature bins and has causal, bounded-context attention. It has no streaming request interface. The diphone branch adapts the published objective to T15 with a frozen GRU and a matched monophone control. The 40-by-40 joint head and separate blank preserve the official initial monophone probabilities. [Time-masked Transformer paper](https://papers.nips.cc/paper_files/paper/2025/file/85e351d89eee3f33d3d1356f38590bb6-Paper-Conference.pdf) · [DCoND paper](https://arxiv.org/html/2411.10657v1)

BEST uses hierarchical phone, character, and audio-compatible representations. The paper reports 1.62% on T15 test with a five-gram and lists 320 GiB in its main table. Our comparison covers the architecture in the paper’s implementation appendix. Runnable weights remain unverified, and the benchmark remains unreproduced here. [BEST paper](https://arxiv.org/html/2610.23175v1)

## Run and resume

Use the verified baseline and decoder environments from [Gilbreth setup](GILBRETH.md). Keep the v2 checkout at `$RCAC_SCRATCH/btt-25-cw/v2` and v1 assets in `workspace`.

After reproducing the v1 feature-only validation cache and training-only three-gram, submit the following on Gilbreth. Set `account` to an allocation authorized for the A100 partition. The repaired arm builds a job-specific wheel and runs its own sustained pilot before decoding.

```bash
p="$RCAC_SCRATCH/btt-25-cw"
mkdir -p "$p/slurm"
slurm=(--account="$account" --partition=a100-40gb --qos=normal --nodes=1 --ntasks=1
       --gpus-per-node=1 --cpus-per-task=4 --mem=15000M --signal=B:USR1@90
       --kill-on-invalid-dep=yes --output="$p/slurm/%x-%j.out" --parsable)
pilot=$(sbatch "${slurm[@]}" --job-name=btt-v2-original-pilot --time=00:24:00 \
  "$p/v2/hpc/gilbreth_v2_pilot.sh")
original=$(sbatch "${slurm[@]}" --job-name=btt-v2-original --time=00:27:00 \
  --dependency="afterok:$pilot" "$p/v2/hpc/gilbreth_v2_search.sh" original "$pilot")
repaired=$(sbatch "${slurm[@]}" --job-name=btt-v2-repaired --time=00:42:00 \
  --dependency="afterok:$original" "$p/v2/hpc/gilbreth_v2_search.sh" repaired "$pilot")
sbatch "${slurm[@]}" --job-name=btt-v2-search-benchmark --time=00:17:00 \
  --dependency="afterok:$repaired" "$p/v2/hpc/gilbreth_v2_search_benchmark.sh" "$repaired"
```

The LightBeam experiment additionally needs its pinned upstream source and the checksum-verified Qwen3-1.7B snapshot. Submit its setup and comparison in sequence after other active experiments.

```bash
setup=$(sbatch "${slurm[@]}" --job-name=btt-v2-lightbeam-setup --time=00:25:00 \
  "$p/v2/hpc/gilbreth_v2_lightbeam_setup.sh")
sbatch "${slurm[@]}" --job-name=btt-v2-lightbeam --time=00:32:00 \
  --dependency="afterok:$setup" "$p/v2/hpc/gilbreth_v2_lightbeam.sh"
```

After the registered search arms finish, pass their condition directories to the fixed-candidate runner. The example below uses the completed three-gram directories. If either registered four-gram arm completes, include its directory too. The runner checks each search registration and chooses the source by the declared tuning rule.

```bash
sources=("$p/v2/runs/v2-search-attempt-$original"/v2-s*/ "$p/v2/runs/v2-search-attempt-$repaired"/v2-s*/)
fixed=$(sbatch "${slurm[@]}" --job-name=btt-v2-fixed --time=00:52:00 \
  "$p/v2/hpc/gilbreth_v2_fixed.sh" "${sources[@]}")
setup4=$(sbatch "${slurm[@]}" --job-name=btt-v2-qwen4-setup --time=00:24:00 \
  --dependency="afterok:$fixed" "$p/v2/hpc/gilbreth_v2_qwen4_setup.sh")
sbatch "${slurm[@]}" --job-name=btt-v2-fixed-4b --time=01:32:00 \
  --dependency="afterok:$setup4" --export="ALL,BTT_QWEN_4B_PRIOR=$p/v2/runs/v2-fixed-$fixed" \
  "$p/v2/hpc/gilbreth_v2_fixed.sh" "${sources[@]}"
```

Retrieve the original campaign's scheduler receipts by job ID.

```bash
p="$RCAC_SCRATCH/btt-25-cw"
sacct -j 12523531,12523569,12523580,12523585,12523601,12523602,12523724,12523725,12523737,12523756,12523757,12523635,12523697,12523698,12523758 \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ReqMem,ExitCode
squeue --me
```

Save one receipt per job and step after each stage. The aggregate exporter reads the columns in the order below.

```bash
sacct -P -j "$completed_job_ids" \
  --format=JobID,State,ElapsedRaw,TotalCPU,UserCPU,SystemCPU,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,AllocTRES,ExitCode \
  > "$receipts_dir/scheduler.psv"
```

The v2 launchers write to a new directory for each job. To resume training directly, run from `workspace` with the v2 `src` on `PYTHONPATH`. Supply the checkpoint and its SHA-256 sidecar, the same condition's `pilot/admission.json`, and the original LM and lexicon.

```bash
"$p/environments/baseline/bin/python" -m btt25.train_v2 \
  --config configs/local.yaml --mode transformer --pilot-seconds 0 --updates 40000 \
  --decoder-python "$p/environments/decoder/bin/python" --lexicon assets/local/lexicon.txt \
  --kenlm-model runs/compact-a100/lm/train-3gram.binary \
  --admission "$pilot/admission.json" --resume "$last_checkpoint" \
  --output "$resume_dir" --max-seconds 5100
```

The reporting job seals the candidate, prediction, configuration, and resource hashes before opening reporting labels. It times the selected systems in three interleaved warm passes, then scores `dev_report` and `official_val`. A timed request runs from opening the released features through final text selection. Cold timing starts fresh processes with filesystem caches intact. The pipeline runs offline, so there is no streaming-latency measurement.

For a budget-stopped Transformer with a valid checkpoint and pilot admission, pass the stopped job ID as the second argument. A new allocation preserves the earlier artifact directory and checks the checkpoint and registered update count.

```bash
sbatch "${slurm[@]}" --job-name=btt-v2-transformer-resume --time=01:40:00 \
  "$p/v2/hpc/gilbreth_v2_encoder.sh" transformer "$stopped_transformer_job"
```

If cgroup charge approaches the target while process memory remains low, check `memory.stat` for file-cache charge. The optional Linux command advises the kernel on cache pages for the manifest's private training HDF5 files. It records memory before and after the call and leaves the data and Slurm limits intact.

```bash
srun --jobid="$transformer_job" --overlap --ntasks=1 --cpus-per-task=1 \
  env PYTHONPATH="$p/v2/src" "$p/environments/baseline/bin/python" -m btt25.cache_advice \
  --manifest "$p/workspace/assets/local/data_manifest.json" \
  --output "$p/v2/runs/v2-e1-transformer-$transformer_job/cache-advice.json"
```

For a hidden-test export, use the original v2 tuning-selected DistilGPT2 system with the same repaired Flashlight build and verified models. The launcher verifies the freeze and manifest, writes phonemes and text from the released features, then saves `submission.csv` in official `id,text` order. Keep the files private. Only coverage and order can be checked.

```bash
sbatch "${slurm[@]}" --job-name=btt-v2-test --time=00:29:00 \
  "$p/v2/hpc/gilbreth_v2_test.sh" "$original_v2_report_job" "$repaired"
```
