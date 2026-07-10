"""Offline head-to-head simulator for bot strategies.

Runs full games by driving Room directly (no redis / HTTP), so hundreds of
games finish in minutes. Useful for measuring win rates between bot
difficulties and tuning strategy parameters.

Usage:
    python simulate.py --bots medium easy --games 100
    python simulate.py --bots medium easy --games 50 --penalty 2 --additional 0
"""
import argparse
import contextlib
import io
import uuid
from collections import defaultdict

from room import Room


def play_game(difficulties: list[str], penalty: int, additional: int) -> dict:
    room = Room.create(
        owner_id="sim-owner",
        owner_name="sim-owner",
        join_code="SIMUL",
        bid_timer=10,
        missing_position_penalty=penalty,
        additional_players_queued=additional,
        nba_era="2025-26"
    )
    # Replace the human owner with bots only. Random ids so auction
    # tie-breaking (dict iteration order) isn't biased toward either bot.
    room.members.clear()
    for difficulty in difficulties:
        room.add_bot(str(uuid.uuid4()), room.default_bot_name(difficulty), difficulty)

    with contextlib.redirect_stdout(io.StringIO()):  # silence bot debug prints
        room.start_game()
        for _ in range(3000):
            if room.round_num <= 0:
                break
            result = room.current_auction.maybe_resolve()
            assert result.resolved, "all-bot auction should resolve immediately"
            room.handle_auction_end(result.winner_id, result.price_paid)
        else:
            raise RuntimeError("game did not terminate")

    return {
        p.name: {
            "score": p.score,
            "spent": 100 - p.balance,
            "team": [n.name for n in p.nba_team],
        }
        for p in room.prev_game_final
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", nargs="+", default=["medium", "easy"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--penalty", type=int, default=1)
    parser.add_argument("--additional", type=int, default=2)
    args = parser.parse_args()

    wins = defaultdict(int)
    score_totals = defaultdict(float)
    spend_totals = defaultdict(float)
    for i in range(args.games):
        final = play_game(args.bots, args.penalty, args.additional)
        winner = max(final, key=lambda name: final[name]["score"])
        wins[winner] += 1
        for name, info in final.items():
            score_totals[name] += info["score"]
            spend_totals[name] += info["spent"]
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{args.games} games, wins so far: {dict(wins)}")

    print(f"\nSettings: penalty={args.penalty} additional={args.additional} "
          f"games={args.games}")
    for name in sorted(score_totals, key=lambda n: -wins[n]):
        print(f"  {name:12s} wins={wins[name]:4d} ({100 * wins[name] / args.games:5.1f}%)  "
              f"avg_score={score_totals[name] / args.games:12.0f}  "
              f"avg_spent={spend_totals[name] / args.games:5.1f}")


if __name__ == "__main__":
    main()
