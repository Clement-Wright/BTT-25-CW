"""Print manuscript tables from the verified rate-study aggregates."""
import argparse
import json
from pathlib import Path


def tables(summary):
    models = summary["models"]
    labels = [
        ("official","Official GRU (80 ms)"), ("gru80","Adapted GRU (80 ms)"),
        ("gru40","Adapted GRU (40 ms)"), ("gru20","Adapted GRU (20 ms)"),
        ("two_rate","80 ms recurrence / 40 ms emissions"), ("patch10","10-bin patch"),
        ("patch20","20-bin patch"), ("bin40","40 ms paired averages"),
        ("bin10_hold","10 ms held grid"), ("adaptive_filter","Adaptive filter"),
        ("activation_only","Softsign/tanh mixture"), ("transform_only","Low-rank input transform"),
        ("input_transform","Transform and activation"), ("causal_conv","Causal convolution")]
    print("| Encoder | Phone edits | PER | Word edits | WER |")
    print("|---|---:|---:|---:|---:|")
    for key, label in labels:
        if key not in models:
            continue
        m = models[key]
        print(f"| {label} | {m['counts']['phone_errors']:,} | {m['per']:.2%} | "
              f"{m['counts']['word_errors']:,} | {m['wer']:.2%} |")
    print("\n| Encoder | P50 | P95 | P99 | Pooled RTF | RSS (GB) | PyTorch GPU peak (GB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key, label in labels:
        if key not in models:
            continue
        t = models[key]["timing"]
        times = [1000*t["seconds"][k] for k in ("p50","p95","p99")]
        print(f"| {label} | {times[0]:.0f} ms | {times[1]:.0f} ms | {times[2]:.0f} ms | "
              f"{t['pooled_rtf']:.4f} | {t['peak_tree_rss_bytes']/1e9:.2f} | "
              f"{t['peak_cuda_allocated_bytes']/1e9:.2f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", nargs="?", type=Path, default=Path("reports/rate_v4.json"))
    tables(json.loads(p.parse_args().summary.read_text()))
