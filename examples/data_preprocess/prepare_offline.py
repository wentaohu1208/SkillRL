"""
Generate dummy parquet files offline without downloading any dataset.
Equivalent to `prepare.py` but with zero network dependency.
"""
import os
import argparse

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='text', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='~/data/verl-agent/')
    parser.add_argument('--train_data_size', default=16, type=int)
    parser.add_argument('--val_data_size', default=64, type=int)
    args = parser.parse_args()

    local_dir = os.path.expanduser(os.path.join(args.local_dir, args.mode))
    os.makedirs(local_dir, exist_ok=True)

    def make_records(size: int, split: str) -> pd.DataFrame:
        records = []
        for idx in range(size):
            records.append({
                "data_source": args.mode,
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": idx},
            })
        return pd.DataFrame(records)

    train_df = make_records(args.train_data_size, "train")
    test_df = make_records(args.val_data_size, "test")

    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Created {train_path} ({args.train_data_size} rows)")
    print(f"Created {test_path} ({args.val_data_size} rows)")
