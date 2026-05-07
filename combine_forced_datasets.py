import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine standard_forced + efficiency_forced into one CSV.")
    ap.add_argument("--standard", default="standard_forced.csv")
    ap.add_argument("--multiprime", default="efficiency_forced.csv")
    ap.add_argument("--out", default="rsa_benchmark_realistic.csv")
    args = ap.parse_args()

    a = pd.read_csv(args.standard)
    b = pd.read_csv(args.multiprime)

    # Basic schema validation
    required = {"scenario", "key_size", "decryption_us", "battery_pct", "cpu_temp_c", "is_charging"}
    for name, df in [("standard", a), ("multiprime", b)]:
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} file missing columns: {sorted(missing)}")

    out = pd.concat([a, b], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"Saved combined dataset: {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()

