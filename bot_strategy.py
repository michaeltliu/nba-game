"""Shared bot bidding strategy.

This module contains the pure bid-computation logic used by the API to drive
in-room bots. It has no console/printing side effects so it can be safely
called from within request handling.

The logic uses a Value Over Replacement Player (VORP) style valuation:

1. Computes the maximum potential team score for each candidate player.
2. Establishes a replacement baseline (average of non-target players).
3. Calculates each player's value over replacement.
4. Allocates the budget across players proportional to their value. This
   ensures we bid on solid players even if they aren't in our absolute
   "dream team", preventing opponents from getting them for free.

The valuation step (1-3) is the expensive part and depends only on the current
roster and the shared player queue -- not on the bidder's balance. It is split
out from the cheap balance-dependent allocation (step 4) so that several bots
sharing an identical roster (e.g. every bot on the first round) can reuse a
single valuation.
"""

import itertools

from game import NBAPlayer


class Valuation:
    """Precomputed VORP valuation for a given roster + candidate queue."""

    def __init__(self, values, current_player_value, sum_target_v, k):
        self.values = values
        self.current_player_value = current_player_value
        self.sum_target_v = sum_target_v
        self.k = k


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


def evaluate(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    current_team: list[NBAPlayer],
) -> Valuation | None:
    """Run the expensive VORP valuation for the player at the front of the
    queue given ``current_team``. Returns ``None`` when no meaningful
    valuation applies (roster full, empty queue, or fewer candidates than
    open slots)."""
    k = 5 - len(current_team)
    if k <= 0 or not player_queue or len(player_queue) <= k:
        return None

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

    # 3. Determine the replacement utility level (average of non-targeted
    # players, excluding the tail that won't realistically be reached).
    end_index = -additional_players if additional_players else None
    replacement_players = sorted_candidates[k:end_index]
    if len(replacement_players):
        u_replacement = sum(utilities[p.pid] for p in replacement_players) / len(replacement_players)
    else:
        u_replacement = 0

    # 4. Value Over Replacement V(P) for each player.
    values = {p.pid: max(0.0, utilities[p.pid] - u_replacement) for p in player_queue}
    sum_target_v = sum(values[p.pid] for p in target_players)

    return Valuation(
        values=values,
        current_player_value=values[current_player.pid],
        sum_target_v=sum_target_v,
        k=k,
    )


def allocate(valuation: Valuation | None, balance: int, player_queue: list[NBAPlayer]) -> int:
    """Turn a (possibly shared) valuation into a concrete bid for a bidder with
    the given balance. Cheap; safe to call once per bot."""
    if not player_queue:
        return 0

    if valuation is None:
        # Either the roster is full (bid 0) or there are no more candidates
        # than open slots -- in the latter case grab the player for $1.
        # Distinguish using team size is not available here, so mirror the
        # original heuristics via compute_bid's guards below.
        return 0

    current_player_value = valuation.current_player_value
    if current_player_value <= 0:
        return 0

    # Allocate budget proportional to value over replacement.
    if valuation.sum_target_v == 0:
        # All target players have 0 value over replacement (e.g., all
        # identical): spend evenly across the open slots.
        bid = max(1, balance // valuation.k)
    else:
        bid = balance * (current_player_value / valuation.sum_target_v)
        bid = int(round(bid))

    # If we want this player (value > 0) and have budget, bid at least $1 so we
    # don't lose to a $0 bid.
    if bid == 0 and balance > 0:
        bid = 1

    # Never exceed our current balance or go below 0.
    return max(0, min(bid, balance))


def compute_bid(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    """Compute a smart bid for the player at the front of the queue.

    Returns an integer bid clamped to the range [0, balance].
    """
    k = 5 - len(current_team)
    if k <= 0:
        return 0
    if not player_queue:
        return 0
    if len(player_queue) <= k:
        return 1

    valuation = evaluate(player_queue, missing_position_penalty, additional_players, current_team)
    return allocate(valuation, balance, player_queue)
