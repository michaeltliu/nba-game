"""Shared bot bidding strategies.

Each difficulty level has its own ``compute_bid`` implementation. They all share
the same signature so the API can dispatch by difficulty:

    compute_bid(difficulty, player_queue, missing_position_penalty,
                additional_players, balance, current_team) -> int

Difficulties (cheapest -> most expensive to compute):

* ``easy``   - naive: spread the remaining budget across open slots, scaled by
               the current player's raw counting stats, with a little jitter.
               No team scoring at all; trivial cost.
* ``medium`` - greedy marginal value: score the current roster plus each single
               candidate and bid proportional to the current player's marginal
               contribution. O(n) team scorings; cheap.
* ``hard``   - Value Over Replacement Player (VORP) with full k-1 lookahead. This
               is the only strategy with combinatorial cost, and a room may hold
               at most one hard bot.

All strategies return an integer bid clamped to ``[0, balance]``.
"""

import itertools
import random

from game import NBAPlayer

DIFFICULTIES = ("easy", "medium", "hard")


def _team_scorer(missing_position_penalty: int):
    # Deferred import to avoid a circular import at module load time
    # (room imports this module).
    from room import Player

    penalty_factor = 1.0 / (2.0 ** (missing_position_penalty / 4.0))

    def get_team_score(team: list[NBAPlayer]) -> float:
        temp_player = Player(name="temp")
        temp_player.nba_team = team
        shortfall = temp_player.best_lineup()
        return temp_player.compute_score(shortfall, penalty_factor)

    return get_team_score


def _proportional_bid(values: dict, current_pid: int, target_players: list, balance: int, k: int) -> int:
    """Shared budget-allocation step used by the medium and hard strategies.

    Bids proportional to the current player's value relative to the sum of the
    top-k target values, falling back to an even split when all targets tie.
    """
    current_value = values[current_pid]
    if current_value <= 0:
        return 0

    sum_target_v = sum(values[p.pid] for p in target_players)
    if sum_target_v == 0:
        is_target = any(p.pid == current_pid for p in target_players)
        bid = max(1, balance // k) if is_target else 0
    else:
        bid = int(round(balance * (current_value / sum_target_v)))

    # If we want this player (value > 0) and have budget, bid at least $1 so we
    # don't lose to a $0 bid.
    if bid == 0 and balance > 0:
        bid = 1

    return max(0, min(bid, balance))


def compute_bid_easy(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue:
        return 0
    if len(player_queue) <= k:
        return min(1, balance)

    current_player = player_queue[0]

    def raw(p: NBAPlayer) -> float:
        return p.pts + p.reb + p.ast + p.stl + p.blk

    max_raw = max((raw(p) for p in player_queue), default=0.0) or 1.0
    desirability = raw(current_player) / max_raw  # ~0..1

    per_slot = balance / k
    bid = int(round(per_slot * desirability * random.uniform(0.5, 1.1)))

    # Occasionally throw in a token $1 for a decent player even if rounding
    # zeroed us out.
    if bid <= 0 and balance > 0 and desirability > 0.4:
        bid = 1

    return max(0, min(bid, balance))


def compute_bid_medium(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue:
        return 0
    if len(player_queue) <= k:
        return min(1, balance)

    current_player = player_queue[0]
    get_team_score = _team_scorer(missing_position_penalty)

    base_score = get_team_score(current_team) if current_team else 0.0
    # Marginal value of adding each candidate to the current roster (no
    # lookahead over how the remaining slots get filled).
    values = {
        p.pid: max(0.0, get_team_score(current_team + [p]) - base_score)
        for p in player_queue
    }

    target_players = sorted(player_queue, key=lambda x: values[x.pid], reverse=True)[:k]
    return _proportional_bid(values, current_player.pid, target_players, balance, k)


def compute_bid_hard(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue:
        return 0
    if len(player_queue) <= k:
        return min(1, balance)

    current_player = player_queue[0]
    get_team_score = _team_scorer(missing_position_penalty)

    # 1. For each candidate player P, compute their utility U(P): the max score
    # achievable if we draft P next and fill the remaining k-1 slots optimally
    # from the rest of the candidates.
    utilities = {}
    for P in player_queue:
        other_players = [x for x in player_queue if x.pid != P.pid]
        target_k_minus_1 = min(k - 1, len(other_players))

        best_score = -1.0
        for subset in itertools.combinations(other_players, target_k_minus_1):
            score = get_team_score(current_team + [P] + list(subset))
            if score > best_score:
                best_score = score
        utilities[P.pid] = best_score

    # 2. Sort candidates by utility descending; the top k are our targets.
    sorted_candidates = sorted(player_queue, key=lambda x: utilities[x.pid], reverse=True)
    target_players = sorted_candidates[:k]

    # 3. Replacement utility level (average of non-targeted players, excluding
    # the tail that won't realistically be reached).
    end_index = -additional_players if additional_players else None
    replacement_players = sorted_candidates[k:end_index]
    if len(replacement_players):
        u_replacement = sum(utilities[p.pid] for p in replacement_players) / len(replacement_players)
    else:
        u_replacement = 0

    # 4. Value Over Replacement, then allocate budget proportionally.
    values = {p.pid: max(0.0, utilities[p.pid] - u_replacement) for p in player_queue}
    return _proportional_bid(values, current_player.pid, target_players, balance, k)


_STRATEGIES = {
    "easy": compute_bid_easy,
    "medium": compute_bid_medium,
    "hard": compute_bid_hard,
}


def compute_bid(
    difficulty: str,
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    """Dispatch to the strategy for ``difficulty``. Unknown difficulties fall
    back to the medium strategy."""
    strategy = _STRATEGIES.get(difficulty, compute_bid_medium)
    return strategy(
        player_queue=player_queue,
        missing_position_penalty=missing_position_penalty,
        additional_players=additional_players,
        balance=balance,
        current_team=current_team,
    )
