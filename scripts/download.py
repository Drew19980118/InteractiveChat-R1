import argparse
import os
import glob
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import orjson

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

# ---- 仓库配置 ----
repo_id = "DrewZhang/qrecc-passages-index"

# ---- 1. 下载 Parquet 分片（6个） ----
print("Downloading parquet shards...")
for i in range(1, 51):
    hf_hub_download(
        repo_id=repo_id,
        filename=f"part_{i:03d}.parquet",
        repo_type="dataset",
        local_dir=args.save_path,
    )

# ---- 2. 下载 E5 索引分片（2个） ----
print("Downloading E5 index shards...")
index_files = [
    "e5_Flat.index.part_aa",
    "e5_Flat.index.part_ab",
    "e5_Flat.index.part_ac",
    "e5_Flat.index.part_ad"
]
for fname in index_files:
    hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        repo_type="dataset",
        local_dir=args.save_path,
    )

# ---- 3. 合并 parquet 为 JSONL（只保留 passage_id 和 passage_text） ----
parquet_files = sorted(glob.glob(os.path.join(args.save_path, "part_*.parquet")))
out_file = os.path.join(args.save_path, "qrecc_index.jsonl")

print(f"Merging {len(parquet_files)} parquet shards into {out_file} ...")
with open(out_file, "wb") as fout:
    for pf in parquet_files:
        pf_reader = pq.ParquetFile(pf)
        for batch in pf_reader.iter_batches(batch_size=50_000, columns=["passage_id", "passage_text"]):
            rows = batch.to_pylist()
            fout.write(b"\n".join(orjson.dumps(r) for r in rows))
            fout.write(b"\n")

print(f"✅ Collection saved to {out_file}")
print(f"✅ Index files saved to {args.save_path}")