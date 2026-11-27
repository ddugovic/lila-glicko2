CATEGORY_REPOS: dict[str, str] = {
    "standard": "Lichess/standard-chess-games",
    "antichess": "Lichess/antichess-chess-games",
    "atomic": "Lichess/atomic-chess-games",
    "chess960": "Lichess/chess960-chess-games",
    "crazyhouse": "Lichess/crazyhouse-chess-games",
    "horde": "Lichess/horde-chess-games",
    "king-of-the-hill": "Lichess/king-of-the-hill-chess-games",
    "racing-kings": "Lichess/racing-kings-chess-games",
    "three-check": "Lichess/three-check-chess-games",
}

VARIANT_CATEGORIES = [c for c in CATEGORY_REPOS if c != "standard"]
STANDARD_SPEEDS = ["bullet", "blitz", "rapid", "classical"]
VALID_SPEEDS = ["none", *STANDARD_SPEEDS]


def speed_where_clause(speed: str) -> str:
    if speed == "none":
        return "AND Event NOT LIKE '%Correspondence%'"
    est_sec = (
        "(TRY_CAST(split_part(TimeControl, '+', 1) AS DOUBLE) "
        "+ 40.0 * COALESCE(TRY_CAST(split_part(TimeControl, '+', 2) AS DOUBLE), 0.0))"
    )
    bounds = {
        "bullet": f"{est_sec} < 180",
        "blitz": f"{est_sec} >= 180 AND {est_sec} < 480",
        "rapid": f"{est_sec} >= 480 AND {est_sec} < 1500",
        "classical": f"{est_sec} >= 1500",
    }[speed]
    return f"AND Event NOT LIKE '%Correspondence%' AND TimeControl LIKE '%+%' AND {bounds}"


def resolve_targets(raw_categories: list[str], raw_speeds: list[str] | None) -> list[tuple[str, str]]:
    cats: list[str] = []
    implicit_targets: list[tuple[str, str]] = []

    for item in raw_categories:
        for raw in item.split(","):
            key = raw.strip().lower()
            if not key:
                continue
            if key == "all":
                cats.extend(c for c in CATEGORY_REPOS if c not in cats)
            elif key == "variants":
                cats.extend(c for c in VARIANT_CATEGORIES if c not in cats)
            elif key in CATEGORY_REPOS:
                if key not in cats:
                    cats.append(key)
            elif key in STANDARD_SPEEDS:
                pair = ("standard", key)
                if pair not in implicit_targets:
                    implicit_targets.append(pair)
            else:
                valid = ", ".join(["all", "variants", *CATEGORY_REPOS, *STANDARD_SPEEDS])
                raise ValueError(f"Unknown category '{raw}'. Valid: {valid}")

    speeds: list[str] = []
    if raw_speeds:
        for item in raw_speeds:
            for raw in item.split(","):
                key = raw.strip().lower()
                if not key:
                    continue
                if key == "all":
                    speeds.extend(s for s in STANDARD_SPEEDS if s not in speeds)
                elif key in VALID_SPEEDS:
                    if key not in speeds:
                        speeds.append(key)
                else:
                    raise ValueError(f"Unknown speed '{raw}'. Valid: all, {', '.join(VALID_SPEEDS)}")

    targets: list[tuple[str, str]] = list(implicit_targets)
    for cat in cats:
        if cat == "standard":
            for s in ([s for s in speeds if s != "none"] or STANDARD_SPEEDS):
                if (cat, s) not in targets:
                    targets.append((cat, s))
        else:
            for s in (speeds or ["none"]):
                if (cat, s) not in targets:
                    targets.append((cat, s))
    return targets


def expand_partitions(partitions: list[str]) -> list[str]:
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
        elif len(start_s) == 7 and len(end_s) == 7 and start_s[4] == "-" and end_s[4] == "-":
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


def make_label(category: str, speed: str, part_tag: str, date_tag: str) -> str:
    cat_slug = category.replace("-", "_")
    base = speed if category == "standard" else (cat_slug if speed == "none" else f"{cat_slug}_{speed}")
    return f"{base}_{part_tag}{date_tag}"

