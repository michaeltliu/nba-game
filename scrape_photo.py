"""Find Basketball Reference headshot URLs by player name.

Usage:
    python scrapephoto.py "Tyrone Hill" "Michael Jordan"
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

BASE_URL = "https://www.basketball-reference.com"
SEARCH_URL = f"{BASE_URL}/search/search.fcgi"
PLAYER_PATH_RE = re.compile(r"^/players/[^/]+/([a-z0-9]+)\.html$")
HEADSHOT_RE = re.compile(r"/images/headshots/[^/?#]+\.jpg(?:[?#].*)?$")


class PlayerSearchParser(HTMLParser):
    """Extract the player link from the first search result."""

    def __init__(self) -> None:
        super().__init__()
        self.in_search_item = False
        self.search_item_depth = 0
        self.player_path: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()

        if tag == "div":
            if self.in_search_item:
                self.search_item_depth += 1
            elif "search-item" in classes:
                self.in_search_item = True
                self.search_item_depth = 1

        if self.in_search_item and tag == "a" and self.player_path is None:
            href = attributes.get("href") or ""
            if PLAYER_PATH_RE.match(href):
                self.player_path = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self.in_search_item:
            self.search_item_depth -= 1
            if self.search_item_depth == 0:
                self.in_search_item = False


class HeadshotParser(HTMLParser):
    """Extract the full-size headshot URL from a player page."""

    def __init__(self) -> None:
        super().__init__()
        self.image_url: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "img" or self.image_url is not None:
            return

        src = dict(attrs).get("src") or ""
        if HEADSHOT_RE.search(src):
            self.image_url = urljoin(BASE_URL, src)


class BasketballReferenceClient:
    def __init__(self, request_delay: float = 3.1) -> None:
        self.request_delay = request_delay
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "nba-game-player-photo-script/1.0 (personal project)"
        )

    def _get(self, url: str, **kwargs: object) -> requests.Response:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        try:
            response = self.session.get(url, timeout=15, **kwargs)
            self.last_request_at = time.monotonic()
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            raise RuntimeError(f"Basketball Reference request failed: {error}") from error

    def find_player(self, name: str) -> tuple[str, str]:
        """Return ``(basketball_reference_id, headshot_url)`` for a name."""
        response = self._get(SEARCH_URL, params={"search": name})

        # Some unambiguous searches redirect straight to the player page.
        player_path = response.url.removeprefix(BASE_URL)
        match = PLAYER_PATH_RE.match(player_path)
        if match is None:
            search_parser = PlayerSearchParser()
            search_parser.feed(response.text)
            player_path = search_parser.player_path or ""
            match = PLAYER_PATH_RE.match(player_path)

        if match is None:
            raise LookupError(f"No Basketball Reference player found for {name!r}")

        player_id = match.group(1)
        player_response = self._get(urljoin(BASE_URL, player_path))
        headshot_parser = HeadshotParser()
        headshot_parser.feed(player_response.text)

        if headshot_parser.image_url is None:
            raise LookupError(
                f"Basketball Reference has no headshot for {name!r} ({player_id})"
            )

        return player_id, headshot_parser.image_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print Basketball Reference IDs and headshot URLs."
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help='a quoted full name, for example "Tyrone Hill"',
    )
    args = parser.parse_args()

    client = BasketballReferenceClient()
    exit_code = 0
    for name in args.names:
        try:
            player_id, image_url = client.find_player(name)
            print(f"{name}\t{player_id}\t{image_url}")
        except (LookupError, RuntimeError) as error:
            print(f"{name}\tERROR\t{error}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())