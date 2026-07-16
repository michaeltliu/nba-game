from pathlib import Path
from typing import Literal

import pandas as pd


DATASET_DIR = Path(
    "~/.cache/kagglehub/datasets/eoinamoore/"
    "historical-nba-data-and-player-box-scores/versions/515"
).expanduser()
STAT_COLUMNS = [
    "points",
    "assists",
    "blocks",
    "steals",
    "reboundsTotal",
    "turnovers",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "threePointersAttempted",
    "threePointersMade",
    "freeThrowsAttempted",
    "freeThrowsMade",
]
POSITION_COLUMNS = ["guard", "forward", "center"]
StatsMode = Literal["era_average", "peak_season"]

# Replace these with the exact regular-season date ranges for your era.
SEASON_DATE_RANGES = [
    ("2000-10-31", "2001-06-30"),
    ("2001-10-30", "2002-06-30"),
    ("2002-10-29", "2003-06-30"),
    ("2003-10-28", "2004-06-30"),
    ("2004-11-02", "2005-06-30"),
    ("2005-11-01", "2006-06-30"),
    ("2006-10-31", "2007-06-30"),
    ("2007-10-30", "2008-06-30"),
    ("2008-10-28", "2009-06-30"),
    ("2009-11-01", "2010-06-30"),
]
STATS_MODE: StatsMode = "peak_season"
MINIMUM_GAMES = 20
OUTPUT_PATH = "player_peaks_2000_10.csv"


def _add_derived_stats(averages: pd.DataFrame) -> pd.DataFrame:
    averages = averages.copy()
    averages["trueShootingAttempts"] = (
        averages["fieldGoalsAttempted"] + 0.44 * averages["freeThrowsAttempted"]
    )
    denominator = 2 * averages["trueShootingAttempts"]
    averages["ts"] = averages["points"].div(denominator.where(denominator.ne(0))).fillna(0.0)
    averages["fantasyPoints"] = (
        averages["points"]
        + 1.5 * averages["reboundsTotal"]
        + 1.5 * averages["assists"]
        + 4 * averages["steals"]
        + 4 * averages["blocks"]
        - 2 * averages["turnovers"]
    )
    return averages


def _aggregate_games(games: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    aggregations = {
        "firstName": "first",
        "lastName": "first",
        "gameId": "count",
        **{column: "mean" for column in STAT_COLUMNS},
    }
    return (
        games.groupby(group_by, as_index=False)
        .agg(aggregations)
        .rename(columns={"gameId": "gamesPlayed"})
    )


def create_player_stats(
    season_ranges: list[tuple[str, str]],
    mode: StatsMode = "era_average",
    minimum_games: int = 20,
    players_path: str | Path = DATASET_DIR / "Players.csv",
    stats_path: str | Path = DATASET_DIR / "PlayerStatistics.csv",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build player averages for an era, optionally using each player's peak season.

    Each season range is ``(start_date, end_date)`` and is inclusive. The
    season label is the calendar year of ``end_date``.
    In ``era_average`` mode, players must average at least ``minimum_games``
    across the seasons in which they played. In ``peak_season`` mode, only
    seasons with at least ``minimum_games`` qualify.
    """
    if mode not in ("era_average", "peak_season"):
        raise ValueError("mode must be 'era_average' or 'peak_season'")
    if minimum_games < 1:
        raise ValueError("minimum_games must be at least 1")
    if not season_ranges:
        raise ValueError("season_ranges must contain at least one season")

    players = pd.read_csv(players_path, usecols=["personId", *POSITION_COLUMNS])
    stats = pd.read_csv(
        stats_path,
        low_memory=False,
        usecols=[
            "firstName",
            "lastName",
            "personId",
            "gameId",
            "gameDate",
            "gameType",
            "numMinutes",
            *STAT_COLUMNS,
        ],
    )
    stats["date"] = pd.to_datetime(stats["gameDate"], errors="coerce")
    stats["numMinutesNumeric"] = pd.to_numeric(stats["numMinutes"], errors="coerce")

    games = stats.loc[
        stats["gameType"].eq("Regular Season") & stats["numMinutesNumeric"].gt(0)
    ].copy()
    games["season"] = pd.NA
    games["seasonOrder"] = pd.NA

    for order, (start_date, end_date) in enumerate(season_ranges):
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        label = str(end.year)
        if start > end:
            raise ValueError(f"season {label} starts after it ends")

        in_season = games["date"].between(start, end)
        if games.loc[in_season, "season"].notna().any():
            raise ValueError(f"season {label} overlaps another season range")
        games.loc[in_season, ["season", "seasonOrder"]] = [label, order]

    games = games[games["season"].notna()].copy()

    if mode == "peak_season":
        averages = _add_derived_stats(
            _aggregate_games(games, ["personId", "season", "seasonOrder"])
        )
        averages = averages[averages["gamesPlayed"] >= minimum_games]
        averages = (
            averages.sort_values(
                ["personId", "fantasyPoints", "gamesPlayed", "seasonOrder"],
                ascending=[True, False, False, False],
            )
            .drop_duplicates("personId", keep="first")
            .copy()
        )
        averages = averages.drop(columns="seasonOrder")
    else:
        averages = _add_derived_stats(_aggregate_games(games, ["personId"]))
        average_games = (
            games.groupby(["personId", "season"])
            .size()
            .groupby("personId")
            .mean()
            .rename("averageGamesPerSeason")
        )
        averages = averages.merge(average_games, on="personId", how="left")
        averages = averages[
            averages["averageGamesPerSeason"] >= minimum_games
        ]

    positions = players[["personId", *POSITION_COLUMNS]].drop_duplicates("personId")
    averages = averages.merge(positions, on="personId", how="left")
    averages[POSITION_COLUMNS] = (
        averages[POSITION_COLUMNS].fillna(0).astype(int)
    )
    averages = averages[averages[POSITION_COLUMNS].any(axis=1)]
    averages["personId"] = averages["personId"].astype(int)
    averages = averages.sort_values("fantasyPoints", ascending=False).reset_index(drop=True)

    columns = list(averages.columns)
    for position in POSITION_COLUMNS:
        columns.remove(position)
    name_end = columns.index("lastName") + 1
    averages = averages[columns[:name_end] + POSITION_COLUMNS + columns[name_end:]]

    if output_path is not None:
        averages.to_csv(output_path, index=False)

    return averages


def main() -> None:
    averages = create_player_stats(
        season_ranges=SEASON_DATE_RANGES,
        mode=STATS_MODE,
        minimum_games=MINIMUM_GAMES,
        output_path=OUTPUT_PATH,
    )
    print(f"Saved stats for {len(averages)} players to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()