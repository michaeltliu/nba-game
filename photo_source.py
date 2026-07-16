"""Build a photo-source report for the highest-ranked players in each CSV."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from player_photos import BasketballReferenceClient

NBA_HEADSHOT_URL = (
    "https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
)
DEFAULT_OUTPUT_NAME = "player_photo_sources.csv"


@dataclass(frozen=True)
class Player:
    pid: str
    name: str


@dataclass
class PlayerPhotoResult:
    player: Player
    has_nba_photo: bool | None = None
    basketball_reference_id: str = ""
    has_basketball_reference_photo: bool | None = None
    check_error: str = ""


def to_rgb_array(img: Image.Image) -> np.ndarray:
    # Palette PNGs with transparency warn if converted straight to RGB;
    # go via RGBA and composite onto white so transparent pixels are stable.
    if img.mode in ("P", "PA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32)


def load_image(source: str) -> np.ndarray:
    """Load an image from a URL or local file path into an RGB array."""
    if source.startswith(("http://", "https://")):
        r = requests.get(source, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(source)
    return to_rgb_array(img)


def same_photo(source1: str, source2: str, max_mean_diff: float = 0) -> bool:
    """True if images are the same size and nearly identical pixel-wise.
    Each source can be a URL or a local file path.
    max_mean_diff: average per-channel difference allowed (0 = exact match).
    """
    a, b = load_image(source1), load_image(source2)
    if a.shape != b.shape:
        return False
    return float(np.mean(np.abs(a - b))) <= max_mean_diff


def find_input_csvs(input_dir: Path) -> list[Path]:
    """Find every player averages and player peaks CSV in a directory."""
    paths = {
        *input_dir.glob("player_averages*.csv"),
        *input_dir.glob("player_peaks*.csv"),
    }
    return sorted(path for path in paths if path.is_file())


def collect_players(csv_paths: list[Path], row_limit: int = 150) -> dict[str, Player]:
    """Collect unique players from the first ``row_limit`` data rows per CSV."""
    players: dict[str, Player] = {}
    required_columns = {"personId", "firstName", "lastName"}

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = required_columns.difference(reader.fieldnames or ())
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{csv_path} is missing required columns: {missing}")

            for row_number, row in enumerate(reader):
                if row_number >= row_limit:
                    break

                pid = row["personId"].strip()
                name = " ".join(
                    part for part in (row["firstName"].strip(), row["lastName"].strip())
                    if part
                )
                if not pid or not name:
                    raise ValueError(
                        f"{csv_path}: row {row_number + 2} has a blank PID or name"
                    )

                existing = players.get(pid)
                if existing is not None and existing.name != name:
                    # NBA data occasionally adds a suffix in later files
                    # (for example, "Jimmy Butler III"). The shorter form is
                    # generally a better Basketball Reference search query.
                    if len(name) < len(existing.name):
                        players[pid] = Player(pid=pid, name=name)
                    continue
                players[pid] = Player(pid=pid, name=name)

    return players


def check_nba_photo(pid: str, blank_photo_path: Path) -> bool:
    """Return false for an NBA placeholder or a definitively missing image."""
    photo_url = NBA_HEADSHOT_URL.format(pid=pid)
    try:
        return not same_photo(photo_url, str(blank_photo_path))
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return False
        raise


def csv_boolean(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def write_report(output_path: Path, results: list[PlayerPhotoResult]) -> None:
    fieldnames = [
        "player_name",
        "pid",
        "has_nba_photo",
        "basketball_reference_id",
        "has_basketball_reference_photo",
        "check_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "player_name": result.player.name,
                    "pid": result.player.pid,
                    "has_nba_photo": csv_boolean(result.has_nba_photo),
                    "basketball_reference_id": (
                        result.basketball_reference_id
                        if result.has_nba_photo is False
                        else ""
                    ),
                    "has_basketball_reference_photo": (
                        csv_boolean(result.has_basketball_reference_photo)
                        if result.has_nba_photo is False
                        else ""
                    ),
                    "check_error": result.check_error,
                }
            )


def build_report(
    input_dir: Path,
    output_path: Path,
    blank_photo_path: Path,
    row_limit: int = 150,
    basketball_reference_delay: float = 3.1,
) -> list[PlayerPhotoResult]:
    csv_paths = find_input_csvs(input_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No player_averages*.csv or player_peaks*.csv files in {input_dir}"
        )
    if not blank_photo_path.is_file():
        raise FileNotFoundError(f"Blank player image not found: {blank_photo_path}")

    players = collect_players(csv_paths, row_limit)
    ordered_players = sorted(players.values(), key=lambda player: player.name.casefold())
    results = [PlayerPhotoResult(player=player) for player in ordered_players]
    players_without_nba_photos: set[str] = set()

    print(
        f"Checking {len(results)} unique players from {len(csv_paths)} CSV files..."
    )
    for index, result in enumerate(results, start=1):
        try:
            result.has_nba_photo = check_nba_photo(
                result.player.pid, blank_photo_path
            )
            if result.has_nba_photo is False:
                players_without_nba_photos.add(result.player.pid)
        except (OSError, requests.RequestException) as error:
            result.check_error = f"NBA photo check failed: {error}"
            print(
                f"[{index}/{len(results)}] {result.player.name}: "
                f"{result.check_error}",
                file=sys.stderr,
            )

    print(
        "Looking up Basketball Reference data for "
        f"{len(players_without_nba_photos)} players without NBA photos..."
    )
    client = BasketballReferenceClient(request_delay=basketball_reference_delay)
    for result in results:
        if result.player.pid not in players_without_nba_photos:
            continue
        try:
            lookup = client.lookup_player(result.player.name)
            result.basketball_reference_id = lookup.player_id
            result.has_basketball_reference_photo = lookup.has_headshot
        except LookupError as error:
            result.has_basketball_reference_photo = False
            result.check_error = str(error)
            print(f"{result.player.name}: {error}", file=sys.stderr)
        except RuntimeError as error:
            result.check_error = str(error)
            print(f"{result.player.name}: {error}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(output_path, results)
    return results


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Check NBA and Basketball Reference photo availability for the "
            "top players in each player averages/peaks CSV."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="directory containing the input CSV files (default: script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT_NAME,
        help=f"report path (default: {DEFAULT_OUTPUT_NAME} beside this script)",
    )
    parser.add_argument(
        "--blank-photo",
        type=Path,
        default=script_dir / "blank_player.png",
        help="NBA placeholder image path (default: blank_player.png beside script)",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=150,
        help="number of data rows read from each CSV (default: 150)",
    )
    parser.add_argument(
        "--br-delay",
        type=float,
        default=3.1,
        help="seconds between Basketball Reference requests (default: 3.1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.row_limit < 1:
        raise ValueError("--row-limit must be at least 1")
    if args.br_delay < 0:
        raise ValueError("--br-delay cannot be negative")

    results = build_report(
        input_dir=args.input_dir.resolve(),
        output_path=args.output.resolve(),
        blank_photo_path=args.blank_photo.resolve(),
        row_limit=args.row_limit,
        basketball_reference_delay=args.br_delay,
    )
    print(f"Wrote {len(results)} players to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())