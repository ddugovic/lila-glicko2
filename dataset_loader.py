from collections import defaultdict
import os
import time
import duckdb
from huggingface_hub import HfApi, hf_hub_download

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
    for item in raw_categories:
        for raw in item.split(","):
            key = raw.strip().lower()
            if not key:
                continue
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


def expand_partitions(partitions: list[str]) -> list[str]:
    """Expands '2013..2015' -> ['2013', '2014', '2015'] and '2015-01..2015-03' -> ['2015-01', '2015-02', '2015-03']."""
    expanded: list[str] = []
    for raw in partitions:
        token = raw.strip()
        if ".." not in token:
            expanded.append(token)
            continue

        start_s, _, end_s = token.partition("..")
        if len(start_s) == 4 and len(end_s) == 4 and start_s.isdigit() and end_s.isdigit():
            y1, y2 = int(start_s), int(end_s)
            if y1 > y2:
                raise ValueError(f"Invalid year range '{token}': start > end.")
            expanded.extend(str(y) for y in range(y1, y2 + 1))
        elif len(start_s) == 7 and len(end_s) == 7 and start_s == "-" and end_s == "-":
            y1, m1 = int(start_s[:4]), int(start_s[5:])
            y2, m2 = int(end_s[:4]), int(end_s[5:])
            if (y1, m1) > (y2, m2) or not (1 <= m1 <= 12 and 1 <= m2 <= 12):
                raise ValueError(f"Invalid month range '{token}'.")
            cur_y, cur_m = y1, m1
            while (cur_y, cur_m) <= (y2, m2):
                expanded.append(f"{cur_y}-{cur_m:02d}")
                cur_m += 1
                if cur_m > 12:
                    cur_m = 1
                    cur_y += 1
        else:
            raise ValueError(f"Invalid range '{token}'. Use YYYY..YYYY or YYYY-MM..YYYY-MM.")
    return expanded


def resolve_partition_prefixes(partitions: list[str]) -> list[str]:
    prefixes: list[str] = []
    for raw in expand_partitions(partitions):
        p = raw.strip()
        if len(p) == 4 and p.isdigit():
            prefixes.append(f"data/year={p}/")
        else:
            year, sep, month = p.partition("-")
            if sep == "-" and len(year) == 4 and year.isdigit() and len(month) in (1, 2) and month.isdigit():
                prefixes.append(f"data/year={year}/month={month.zfill(2)}/")
            else:
                raise ValueError(f"Invalid partition '{raw}'. Use YYYY, YYYY-MM, YYYY..YYYY, or YYYY-MM..YYYY-MM.")
    return prefixes


def prepare_datasets(
    categories: list[str],
    partitions: list[str],
    part_tag: str,
    date_tag: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    hf_token: str | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Groups requested categories by HF repo_id so shared shards (such as standard-chess-games)
    are downloaded once 1-by-1, extracted into isolated per-category Parquet files, and deleted
    immediately to minimize disk usage and Hugging Face resolver requests.
    Returns {category: (label, parquet_path)}.
    """
    ready: dict[str, tuple[str, str]] = {}
    by_repo: dict[str, list[tuple[str, str, str, str | None]]] = defaultdict(list)

    for cat in categories:
        label = f"{cat.replace('-', '_')}_{part_tag}{date_tag}"
        out_path = f"{label}_rd_tuning.parquet"
        if not refresh and os.path.exists(out_path):
            print(f"Using cached Parquet file: {out_path}")
            ready[cat] = (label, out_path)
            continue
        if cat == "rapid" and all(int(p[:4]) <= 2017 for p in partitions):
            print(f"  Skipping rapid: 0 games exist before Dec 2, 2017 (partitions={partitions}).")
            continue
        repo_id, event_filter = CATEGORIES[cat]
        by_repo[repo_id].append((cat, label, out_path, event_filter))

    if not by_repo:
        return ready

    prefixes = resolve_partition_prefixes(partitions)
    api = HfApi(token=hf_token)
    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_spill';")

    base_where = [
        "Result IN ('1-0', '0-1', '1/2-1/2')",
        "White IS NOT NULL",
        "Black IS NOT NULL",
        "WhiteRatingDiff IS NOT NULL",
        "BlackRatingDiff IS NOT NULL",
        "UTCDate IS NOT NULL",
        "UTCTime IS NOT NULL",
    ]
    if start_date:
        base_where.append(f"UTCDate >= DATE '{start_date}'")
    if end_date:
        base_where.append(f"UTCDate <= DATE '{end_date}'")
    base_where_sql = " AND ".join(base_where)

    for repo_id, targets in by_repo.items():
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        shard_paths: list[str] = []
        for prefix in prefixes:
            matched = [f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")]
            if matched:
                shard_paths.extend(matched)

        if not shard_paths:
            print(f"  Skipping {', '.join(c for c, _, _, _ in targets)}: no shards found in {repo_id}.")
            continue

        cat_names = ", ".join(c for c, _, _, _ in targets)
        print(f"Streaming {len(shard_paths)} shard(s) 1-by-1 from {repo_id} for [{cat_names}]...")
        t0 = time.perf_counter()
        parts_by_cat: dict[str, list[str]] = defaultdict(list)

        try:
            for idx, rel_path in enumerate(shard_paths, 1):
                print(f"  [{idx}/{len(shard_paths)}] Downloading {rel_path}...")
                local_raw = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path, token=hf_token)
                try:
                    for cat, _, _, event_filter in targets:
                        part_path = f".tmp_{cat}_{idx:04d}.parquet"
                        speed_sql = f"AND Event LIKE '%{event_filter}%'" if event_filter else ""
                        con.execute(f"""
                        COPY (
                            SELECT
                                White AS white, Black AS black,
                                CASE Result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 WHEN '1/2-1/2' THEN 0.5 END AS score,
                                epoch(UTCDate::DATE + UTCTime::TIME) AS timestamp,
                                WhiteElo AS white_elo, BlackElo AS black_elo, TimeControl AS time_control
                            FROM read_parquet('{local_raw}')
                            WHERE {base_where_sql} {speed_sql}
                        ) TO '{part_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                        """)
                        parts_by_cat[cat].append(part_path)
                finally:
                    real_blob = os.path.realpath(local_raw)
                    if os.path.exists(local_raw):
                        os.remove(local_raw)
                    if os.path.exists(real_blob):
                        os.remove(real_blob)

            for cat, label, out_path, _ in targets:
                con.execute(f"""
                COPY (SELECT * FROM read_parquet({parts_by_cat[cat]}) ORDER BY timestamp ASC)
                TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
                count, first_ts, last_ts = con.execute(
                    f"SELECT COUNT(*), MIN(to_timestamp(timestamp)), MAX(to_timestamp(timestamp)) FROM '{out_path}'"
                ).fetchone()
                if count == 0:
                    os.remove(out_path)
                    print(f"  Skipping {cat}: 0 games matched.")
                else:
                    print(f"  Saved {count:,} {cat} games ({first_ts} to {last_ts})")
                    ready[cat] = (label, out_path)
        finally:
            for plist in parts_by_cat.values():
                for p in plist:
                    if os.path.exists(p):
                        os.remove(p)

        print(f"Finished {repo_id} extraction in {time.perf_counter() - t0:.2f}s")

    return ready

