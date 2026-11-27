from collections import defaultdict
import os
import time
import duckdb
from huggingface_hub import HfApi, hf_hub_download
from partitions import CATEGORY_REPOS, make_label, resolve_partition_prefixes, speed_where_clause


def list_available_partitions(targets: list[tuple[str, str]], hf_token: str | None = None):
    """Queries Hugging Face repo file trees and prints available years/months per unique dataset repo."""
    api = HfApi(token=hf_token)
    seen_repos: dict[str, list[str]] = defaultdict(list)
    for cat, _ in targets:
        repo_id = CATEGORY_REPOS[cat]
        if cat not in seen_repos[repo_id]:
            seen_repos[repo_id].append(cat)

    for repo_id, cats in seen_repos.items():
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        months_by_year: dict[str, set[str]] = defaultdict(set)
        for path in repo_files:
            if not path.startswith("data/year=") or not path.endswith(".parquet"):
                continue
            parts = path.split("/")
            year = parts[1].removeprefix("year=")
            month = parts[2].removeprefix("month=") if len(parts) > 2 and parts[2].startswith("month=") else None
            if month:
                months_by_year[year].add(month)

        print(f"\n=== {repo_id} ({', '.join(cats)}) ===")
        for year in sorted(months_by_year):
            months = sorted(months_by_year[year])
            status = "complete (01..12)" if len(months) == 12 else f"partial ({', '.join(months)})"
            print(f"  {year}: {status}")


def prepare_datasets(
    targets: list[tuple[str, str]],
    partitions: list[str],
    part_tag: str,
    date_tag: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    hf_token: str | None = None,
) -> dict[tuple[str, str], tuple[str, str]]:
    ready: dict[tuple[str, str], tuple[str, str]] = {}
    by_repo: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    for cat, speed in targets:
        label = make_label(cat, speed, part_tag, date_tag)
        out_path = f"{label}_rd_tuning.parquet"
        if not refresh and os.path.exists(out_path):
            print(f"Using cached Parquet file: {out_path}")
            ready[(cat, speed)] = (label, out_path)
            continue
        by_repo[CATEGORY_REPOS[cat]].append((cat, speed, label, out_path))

    if not by_repo:
        return ready

    prefixes = resolve_partition_prefixes(partitions)
    api = HfApi(token=hf_token)
    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")

    base_where = [
        "Result IN ('1-0', '0-1', '1/2-1/2')",
        "White IS NOT NULL", "Black IS NOT NULL",
        "WhiteRatingDiff IS NOT NULL", "BlackRatingDiff IS NOT NULL",
        "UTCDate IS NOT NULL", "UTCTime IS NOT NULL",
    ]
    if start_date:
        base_where.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        base_where.append(f"UTCDate <= DATE '{end_date}'")
    base_where_sql = " AND ".join(base_where)

    for repo_id, repo_targets in by_repo.items():
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        shard_paths = [
            f for prefix in prefixes for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")
        ]
        if not shard_paths:
            print(f"  Skipping {', '.join(f'{c}/{s}' for c, s, _, _ in repo_targets)}: no shards found in {repo_id}.")
            continue

        desc = ", ".join(f"{c}/{s}" for c, s, _, _ in repo_targets)
        print(f"Streaming {len(shard_paths)} shard(s) 1-by-1 from {repo_id} for [{desc}]...")
        t0 = time.perf_counter()
        parts_by_target: dict[tuple[str, str], list[str]] = defaultdict(list)

        try:
            for idx, rel_path in enumerate(shard_paths, 1):
                print(f"  [{idx}/{len(shard_paths)}] Downloading {rel_path}...")
                local_raw = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path, token=hf_token)
                try:
                    for cat, speed, _, _ in repo_targets:
                        part_path = f".tmp_{cat}_{speed}_{idx:04d}.parquet"
                        con.execute(f"""
                        COPY (
                            SELECT
                                White AS white, Black AS black,
                                CASE Result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 WHEN '1/2-1/2' THEN 0.5 END AS score,
                                epoch(UTCDate::DATE + UTCTime::TIME) AS timestamp,
                                WhiteElo AS white_elo, BlackElo AS black_elo, TimeControl AS time_control
                            FROM read_parquet('{local_raw}')
                            WHERE {base_where_sql} {speed_where_clause(speed)}
                        ) TO '{part_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                        """)
                        parts_by_target[(cat, speed)].append(part_path)
                finally:
                    real_blob = os.path.realpath(local_raw)
                    if os.path.exists(local_raw):
                        os.remove(local_raw)
                    if os.path.exists(real_blob):
                        os.remove(real_blob)

            for cat, speed, label, out_path in repo_targets:
                con.execute(f"""
                COPY (SELECT * FROM read_parquet({parts_by_target[(cat, speed)]}) ORDER BY timestamp ASC)
                TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
                count, first_ts, last_ts = con.execute(
                    f"SELECT COUNT(*), MIN(to_timestamp(timestamp)), MAX(to_timestamp(timestamp)) FROM '{out_path}'"
                ).fetchone()
                if count == 0:
                    os.remove(out_path)
                    print(f"  Skipping {cat}/{speed}: 0 games matched.")
                else:
                    print(f"  Saved {count:,} {cat}/{speed} games ({first_ts} to {last_ts})")
                    ready[(cat, speed)] = (label, out_path)
        finally:
            for plist in parts_by_target.values():
                for p in plist:
                    if os.path.exists(p):
                        os.remove(p)

        print(f"Finished {repo_id} extraction in {time.perf_counter() - t0:.2f}s")

    return ready

