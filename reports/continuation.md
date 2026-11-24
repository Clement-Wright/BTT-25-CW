# Accuracy and efficiency continuation

The frozen v1 analysis remains available at its [release revision](https://github.com/Clement-Wright/BTT-25-CW/blob/fe5fb4081f2ce256b41ba2a91f104a6a64a53deb/reports/results.md). Its every-trial DistilGPT2 condition scored 17.67% WER on the 500-trial `dev_report` selection and 18.21% on full validation. Those results informed this continuation. V2 development uses the same 926-trial `dev_tune` partition. Any v2 report score on the existing benchmark will be exploratory.

## Frozen v2 results

The tuning rule selected repaired beam-250 Flashlight followed by pinned DistilGPT2 reranking of up to 25 saved candidates. It made **505 / 3,412 word edits (14.80% WER)** on the 500-trial `dev_report` benchmark and **1,307 / 9,445 (13.84%)** on all 1,426 validation trials. These results are exploratory because v1 reporting results informed v2. The selection and artifact hashes were sealed before v2 opened reporting references. The freeze is `db4bae4fbcd3783cbdd55b50bfb5569002db7eeef5ded7c51db54cc867fef9b8`.

| Condition | `dev_tune` WER | Exploratory `dev_report` WER | Exploratory full-validation WER |
| --- | ---: | ---: | ---: |
| Original compact beam 50 | 20.45% | 19.87% | 20.24% |
| Repaired compact beam 250 | 15.80% | 16.74% | 16.14% |
| Repaired + DistilGPT2, tuning-selected | 13.29% | 14.80% | 13.84% |
| Repaired + Qwen3-1.7B-Base | 13.49% | 15.01% | 14.04% |
| Repaired + Qwen3-4B-Base | 13.41% | 14.74% | 13.89% |
| Monophone control, original compact beam 50 | 20.60% | 19.93% | 20.36% |
| Joint diphone adaptation, original compact beam 50 | 23.39% | 22.63% | 23.11% |

All frozen conditions covered all 1,426 validation trials with zero failed predictions. The official GRU’s report PER stayed at 10.54% across decoder conditions. The selected system cut 173 report edits versus original compact, a change of −5.07 percentage points with a paired session-block bootstrap interval of −6.26 to −3.94 points. It improved 153 trials and worsened 30. Reaching 10% requires 164 more edits removed on `dev_report`, or 363 on full validation. The sealed package includes edit types, paired intervals, session/corpus totals, and fixed examples. Reference-bearing examples stay with the private run. [Aggregate evidence](continuation.json)

### Fixed-candidate scoring

All three neural models scored the same candidates with Flashlight’s acoustic component and a teacher-forced language prior. Each chose weight 1.25 on tuning, the largest registered value. The Qwen 4B arm had the lowest observed exploratory `dev_report` WER by two edits over DistilGPT2. DistilGPT2 led on tuning and on full validation. Qwen 4B versus Qwen 1.7B changed `dev_report` WER by −0.26 points, with a paired session-block interval of −0.76 to +0.20 points. Qwen 4B saved nine Switchboard edits versus Qwen 1.7B and added two on OpenWebText. The reporting split contains no random-word corpus. On the 200 random-word tuning tokens, repaired compact made 76 errors, versus 82 with DistilGPT2, 78 with Qwen 1.7B, and 81 with Qwen 4B.

Across full-validation candidate scoring, cached-candidate p95 was 34 ms for DistilGPT2, 150 ms for Qwen 1.7B, and 188 ms for Qwen 4B. These are incremental scoring times. The first two shared one Slurm allocation. Qwen 4B used a separate allocation with the same requested resources and scored the identical candidate file. The two jobs peaked at 5.66 and 10.08 GB device-wide, respectively. The observed scores offer little reason to move to an 8B reranker on these lists.

### Matched end-to-end timing after the freeze

One PCIe A100 40 GB allocation with four CPUs completed 1,152 feature-to-text requests over three interleaved warm passes, with zero failures or timeouts. Every row used the same 96 trials. The real-time factor uses the original input duration.

| System | p50 | p90 | p95 | p99 | Max | Real-time factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original compact beam 50 | 59.3 ms | 91.6 ms | 103.9 ms | 128.7 ms | 157.1 ms | 0.00352 |
| Repaired compact beam 250 | 110.6 ms | 228.7 ms | 250.4 ms | 285.5 ms | 333.6 ms | 0.00707 |
| Repaired + DistilGPT2 | 140.6 ms | 260.3 ms | 284.7 ms | 355.2 ms | 783.6 ms | 0.00863 |
| Monophone control | 58.7 ms | 91.3 ms | 102.3 ms | 125.7 ms | 166.2 ms | 0.00349 |

![Exploratory WER and complete-pipeline latency on one A100](figures/v2_accuracy_latency.png)

The tuning-selected system completed 6.33 requests per second of request time. Its measured cold DistilGPT2 setup took 61.71 s and native decoder readiness took 2.79 s. The shared benchmark harness peaked at 2.57 GB cgroup memory, 2.29 GB process-tree RSS, 2.21 GB process-tree PSS, and 1.95 GB device-wide GPU memory. These peaks cover all resident workers. Streaming latency is unsupported. [Timing distributions and accounting](continuation.json)

### Alternative encoder objectives

The matched 1,024-update frozen-GRU experiment found that the joint diphone head worsened report WER by 2.70 points versus the monophone control. The paired session-block interval was +1.90 to +3.40 points. It improved 49 trials and worsened 121. Report PER rose from 10.66% to 11.72%. Both arms kept the same original compact decoder. This tests a T15 objective adaptation, while the published DCoND-LIFT system used T12 with additional stages.

The scratch-trained Transformer passed its 901-second pilot with 21,466 updates. Its first training continuation stopped at update 24,987 with a valid checkpoint when cgroup charge reached 12.014 GB. Process-tree RSS peaked at 1.54 GB. A new allocation resumes the same checkpoint and the registered 40,000-update endpoint under the same host limits. The first v2 freeze excludes the stopped attempt.

## Candidate coverage

The first comparison holds the official GRU emissions, decoder thresholds, lexicon, and training-only KenLM weights fixed. Each new arm saved all 1,426 validation hypotheses before scoring the tuning slice. Oracle cells use rankings from the actual distinct candidates returned. A failed or empty hypothesis would contribute its full word deletions to the corpus count.

| Decoder | Selected edits | Selected WER | Oracle@10 | Oracle@25 | Oracle@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived beam 50, 10 requested | 1,234 | 20.45% | 16.34% | 16.34% | 16.34% |
| Original beam 50, 50 requested | 1,234 | 20.45% | 16.34% | 15.65% | 15.61% |
| Original beam 100, 50 requested | 1,163 | 19.28% | 15.12% | 14.16% | 13.77% |
| Original beam 250, 50 requested | 1,101 | 18.25% | 14.01% | 12.78% | 12.02% |
| Repaired beam 50, 50 requested | 1,068 | 17.70% | 13.36% | 12.68% | 12.66% |
| Repaired beam 100, 50 requested | 1,011 | 16.76% | 12.28% | 11.27% | 10.92% |
| Repaired beam 250, 50 requested | 953 | 15.80% | 11.22% | 10.00% | 9.38% |

Each row uses 6,033 reference words. The original beam-250 row made 217 fewer oracle errors than the new original beam-50 list. The repair at beam 250 cut selected errors by another 148 and oracle errors by 159. Its top 50 oracle makes 566 edits, which clears the 10% threshold by 37 edits. The 387-edit ranking gap to the selected output is specific to this tuning set and its saved list. The archived list has at most ten candidates, so all higher oracle cutoffs reuse that list.

![Selected and oracle WER by search setting](figures/v2_candidate_coverage.png)

The native trie audit found that a six-label terminal cap dropped 224 pronunciation entries and made 208 word types unreachable. Nine of those types appeared 1,414 times in training, including 1,324 instances of `you`. The repair retained all 134,711 pronunciations and reduced tuning lexicon OOVs from 144 to 3 word tokens. KenLM OOVs remained at 402 tokens because the three-gram stayed fixed. [Registration and commands](../docs/CONTINUATION.md)

At beam 250, the largest change came from Switchboard with 130 fewer errors over 4,569 words. Random-word trials tied at 76 errors over 200 words. Harvard sentences had three additional errors over 201 words. These corpus slices share the 926-trial tuning scope and the same neural logits. [Corpus and session counts](continuation.json)

The external four-gram downloaded and decompressed in seven seconds. Direct KenLM membership checks matched the publisher’s companion vocabulary across all 125,924 decoder dictionary words, with 69,843 shared words. The original beam-100 and repaired beam-250 four-gram pilots stopped after 8,384 and 4,867 successful requests. In each run, one native request stalled for 30 seconds after about ten warm minutes. The timed-out trials differed and had previously completed 87 and 50 times, with maximum times of 181 and 154 ms. Cgroup peaks were 0.74 and 0.91 GB, with zero OOM events. Neither run met sustained admission or produced a tuning WER. A future reproduction needs native service liveness and IPC tracing around the ten-minute mark. [Partial timing and scheduler receipts](continuation.json)

## Complete requests on the laptop

The laptop benchmark used the official encoder on an RTX 3080 Laptop GPU and the Windows-compatible ZeroLM lexicon decoder. It timed released feature reads, transfer, smoothing, encoder completion, native search, and final text selection. Across three interleaved warm passes, each beam completed 288 requests on the same 96 length-sampled tuning trials. Batch size was one. References stayed closed, and the benchmark ran offline.

| ZeroLM beam | p50 | p90 | p95 | p99 | Max | Real-time factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 66.7 ms | 105.2 ms | 127.9 ms | 146.1 ms | 158.1 ms | 0.00392 |
| 100 | 96.2 ms | 152.5 ms | 184.5 ms | 219.2 ms | 238.0 ms | 0.00572 |

Combined process-tree RSS peaked at 1.53 GB with both decoder processes resident. Device-wide use peaked at 2.28 GB. The encoder process reported a 0.37 GB PyTorch allocated peak and 1.07 GB reserved peak. Cold checkpoint setup took 1.11 s. The two native services became ready in 2.11 s and 2.02 s. ZeroLM timing describes that laptop configuration. The matched KenLM and neural-LM systems use the Gilbreth allocations.

The original beam-250 cached-logit job charged up to 1.35 GB to its job cgroup, with process RSS around 0.28 GB. Its 15,000 MiB Slurm request set an effective host limit of 15,728,640,000 bytes. Cgroup charge includes the file cache and is reported separately from process RSS. [Aggregate counts and timings](continuation.json)

## Matched complete-pipeline timing on Gilbreth

Three interleaved warm passes timed the same 96 feature trials across three search configurations on one PCIe A100 40 GB allocation with four CPUs. The encoder ran in FP32 and synchronized each GPU phase. All 864 requests returned text. The real-time factor divides total completed request time by original input duration in 20 ms bins.

| System | p50 | p90 | p95 | p99 | Max | Real-time factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original beam 50 | 64.7 ms | 135.5 ms | 222.4 ms | 312.1 ms | 488.0 ms | 0.00446 |
| Original beam 250 | 109.3 ms | 220.0 ms | 243.3 ms | 263.4 ms | 320.8 ms | 0.00699 |
| Repaired beam 250 | 111.3 ms | 230.7 ms | 248.3 ms | 298.6 ms | 313.7 ms | 0.00706 |

![Matched complete-pipeline latency distribution](figures/v2_search_latency.png)

Repaired beam 250 completed 7.74 requests per second of request time. Cold encoder initialization took 6.26 s within the Python benchmark. The combined allocation with three resident decoder services peaked at 3.11 GB cgroup charge, 1.41 GB summed RSS, 1.35 GB PSS, and 1.61 GB device-wide GPU use. The PyTorch process peaked at 0.36 GB allocated and 1.07 GB reserved. The cgroup used 99.90 user and 12.05 system CPU core-seconds, averaging 0.86 busy cores out of four. The shared-resource figures cover the whole harness and cannot be assigned to individual arms. [Length-stratified timings and counters](continuation.json)

## LightBeam with in-search fusion

The pinned LightBeam run decoded a fixed length-stratified subset of 24 `dev_tune` trials. It saved text and phones for every trial in each arm. Twenty passes completed 1,440 decoder requests in 940.55 warm seconds. Only the first pass contributed to word errors. Qwen3-1.7B-Base scored exact candidate tokens inside the Gilbreth A100 allocation. This uses the same final scorer in both neural arms and the published search backend with our own settings and assets. [LightBeam paper](https://arxiv.org/html/2603.14002v1)

| Condition | Word edits / 156 | WER |
| --- | ---: | ---: |
| LightBeam with training-only three-gram | 31 | 19.87% |
| LightBeam with final-only Qwen | 30 | 19.23% |
| LightBeam with periodic and final Qwen | 23 | 14.74% |
| Original compact beam 50 on the same trials | 25 | 16.03% |
| Repaired compact beam 250 on the same trials | 23 | 14.74% |

Periodic fusion had nine fewer deletions and two more substitutions than final-only scoring. Three trials improved and one worsened. Repaired compact made 21 substitutions and two insertions. Periodic LightBeam made 20 substitutions and three deletions. The subset result does not establish a broader WER improvement.

The periodic arm scored 527,000 target tokens across the timing passes, compared with 50,220 for final-only. Its cached-logit p95 was about 1.98 s, versus 0.79 s for final-only. The job used an SXM4 A100 40 GB and peaked at 2.77 GB of cgroup charge, 4.12 GB of summed process RSS and 4.81 GB of device-wide GPU use. Shared memory can appear more than once in RSS. Cold initialization took 121.75 s. The periodic arm matched the simpler repaired search on this common subset, so this configuration remains a feasibility study. [LightBeam measurements](continuation.json)
