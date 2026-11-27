import os
import time
import duckdb
from huggingface_hub import HfApi, hf_hub_download

# Maps every Lichess rating category to (hf_repo_id, optional_event_filter)
CATEGORIES: dict[str, tuple[str, str | None]] = {
    "bullet": ("Lichess/standard-chess-games", "Bullet"),
    "blitz": ("Lichess/standard-chess-games", "Blitz"),
    "rapid": ("Lichess/standard-chess-games", "Rapid"),
    "classical": ("Lichess/standard-chess-games", "Classical"),
    "antichess": ("Lichess/antichess-chess-games", None),
    "atomic": ("Lichess/atomic-chess-games", None),
    "chess960": ("Lichess/chess960-chess-games", None),
    "crazyhouse": ("Lichess/crazyhouse-chess-games", None),
    "horde": ("Lichess/horde-chess-games", None),
    "king-of-the-hill": ("Lichess/king-of-the-hill-chess-games", None),
    "racing-kings": ("Lichess/racing-kings-chess-games", None),
    "three-check": ("Lichess/three-check-chess-games", None),
}


def resolve_categories(raw_categories: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw in raw_categories:
        key = raw.strip().lower()
        if key == "all":
            for c in CATEGORIES:
                if c not in resolved:
                    resolved.append(c)
        elif key in CATEGORIES:
            if key not in resolved:
                resolved.append(key)
        else:
            valid = ", ".join(["all", *CATEGORIES.keys()])
            raise ValueError(f"Unknown category '{raw}'. Valid choices: {valid}")
    return resolved


def resolve_partition_prefixes(partitions: list[str]) -> list[str]:
    prefixes: list[str] = []
    for raw in partitions:
        p = raw.strip()
        if len(p) == 4 and p.isdigit():
            prefixes.append(f"data/year={p}/")
        else:
            year, sep, month = p.partition("-")
            if sep == "-" and len(year) == 4 and year.isdigit() and len(month) in (1, 2) and month.isdigit():
                prefixes.append(f"data/year={year}/month={month.zfill(2)}/")
            else:
                raise ValueError(f"Invalid partition '{raw}'. Use YYYY (e.g. 2015) or YYYY-MM (e.g. 2015-01).")
    return prefixes


def prepare_dataset(
    category: str,
    partitions: list[str],
    label: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    hf_token: str | None = None,
) -> str | None:
    out_path = f"{label}_rd_tuning.parquet"
    if not refresh and os.path.exists(out_path):
        print(f"Using cached Parquet file: {out_path}")
        return out_path

    repo_id, event_filter = CATEGORIES[category]

    if category == "rapid" and all(int(p[:4]) <= 2017 for p in partitions):
        print(f"  Skipping rapid: 0 games exist before Dec 2, 2017 (partitions={partitions}).")
        return None

    prefixes = resolve_partition_prefixes(partitions)
    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    shard_paths: list[str] = []
    for prefix in prefixes:
        matched = [f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")]
        if not matched:
            print(f"  Skipping {category}: no shards under {prefix} in {repo_id}.")
            return None
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
    if event_filter:
        where_clauses.append(f"Event LIKE '%{event_filter}%'")
    if start_date:
        where_clauses.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"UTCDate <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)
    print(f"Streaming {len(shard_paths)} shard(s) 1-by-1 for {category} ({', '.join(partitions)}) -> {out_path}...")
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
        print(f"  Skipping {category}: 0 games matched in {repo_id} (partitions={partitions}).")
        return None

    print(f"  Saved {count:,} {category} games ({first_ts} to {last_ts}) in {time.perf_counter() - t0:.2f}s")
    return out_path

