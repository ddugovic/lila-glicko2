import os
import time
import duckdb
from huggingface_hub import HfApi, hf_hub_download


def resolve_partition_prefixes(partitions: list[str]) -> list[str]:
    """Converts ['2015'] -> ['data/year=2015/'] and ['2015-01'] -> ['data/year=2015/month=01/']."""
    prefixes: list[str] = []
    for raw in partitions:
        p = raw.strip()
        if len(p) == 4 and p.isdigit():
            prefixes.append(f"data/year={p}/")
        elif len(p) in (6, 7) and p[4] == "-" and p[:4].isdigit() and p[5:].isdigit():
            prefixes.append(f"data/year={p[:4]}/month={p[5:].zfill(2)}/")
        else:
            raise ValueError(f"Invalid partition '{raw}'. Use YYYY (e.g. 2015) or YYYY-MM (e.g. 2015-01).")
    return prefixes


def prepare_dataset(
    repo_id: str,
    partitions: list[str],
    speed: str | None,
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    hf_token: str | None = None,
) -> str | None:
    """Downloads shards 1-by-1 via hf_hub_download and extracts an isolated RD-tuning Parquet file."""
    out_path = f"{label}_rd_tuning.parquet"
    if not refresh and os.path.exists(out_path):
        print(f"Using cached Parquet file: {out_path}")
        return out_path

    if speed == "Rapid" and all(int(p[:4]) <= 2017 for p in partitions):
        print(f"  Skipping Rapid: 0 games exist before Dec 2, 2017 (partitions={partitions}).")
        return None

    prefixes = resolve_partition_prefixes(partitions)
    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    shard_paths: list[str] = []
    for prefix in prefixes:
        matched = [f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")]
        if not matched:
            raise FileNotFoundError(f"No Parquet shards found under {prefix} in {repo_id}")
        shard_paths.extend(matched)

    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")

    where_clauses = [
        "Result IN ('1-0', '0-1', '1/2-1/2')",
        "White IS NOT NULL",
        "Black IS NOT NULL",
        "WhiteRatingDiff IS NOT NULL",
        "BlackRatingDiff IS NOT NULL",
        "UTCDate IS NOT NULL",
        "UTCTime IS NOT NULL",
    ]
    if speed:
        where_clauses.append(f"Event LIKE '%{speed}%'")
    if start_date:
        where_clauses.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"UTCDate <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)
    pool_desc = speed or repo_id.split("/")[-1]
    print(f"Streaming {len(shard_paths)} shard(s) 1-by-1 for {pool_desc} ({', '.join(partitions)}) -> {out_path}...")
    t0 = time.perf_counter()

    compact_parts: list[str] = []
    try:
        for idx, rel_path in enumerate(shard_paths, 1):
            print(f"  [{idx}/{len(shard_paths)}] Downloading {rel_path}...")
            local_raw = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path, token=hf_token)
            part_path = f".tmp_{label}_part_{idx:04d}.parquet"
            try:
                con.execute(f"""
                COPY (
                    SELECT
                        White AS white, Black AS black,
                        CASE Result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 WHEN '1/2-1/2' THEN 0.5 END AS score,
                        epoch(UTCDate::DATE + UTCTime::TIME) AS timestamp,
                        WhiteElo AS white_elo, BlackElo AS black_elo, TimeControl AS time_control
                    FROM read_parquet('{local_raw}')
                    WHERE {where_sql}
                ) TO '{part_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
                compact_parts.append(part_path)
            finally:
                real_blob = os.path.realpath(local_raw)
                if os.path.exists(local_raw):
                    os.remove(local_raw)
                if os.path.exists(real_blob):
                    os.remove(real_blob)

        con.execute(f"""
        COPY (SELECT * FROM read_parquet({compact_parts}) ORDER BY timestamp ASC)
        TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    finally:
        for p in compact_parts:
            if os.path.exists(p):
                os.remove(p)

    count, first_ts, last_ts = con.execute(
        f"SELECT COUNT(*), MIN(to_timestamp(timestamp)), MAX(to_timestamp(timestamp)) FROM '{out_path}'"
    ).fetchone()

    if count == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"  Skipping {pool_desc}: 0 games matched in {repo_id} (partitions={partitions}).")
        return None

    print(f"  Saved {count:,} {pool_desc} games ({first_ts} to {last_ts}) in {time.perf_counter() - t0:.2f}s")
    return out_path

