"""Shared bot bidding strategies.

Each difficulty level has its own ``compute_bid`` implementation.
"""

import itertools
from game import NBAPlayer
from player import Player, _REQUIRED_POS_COUNTS

DIFFICULTIES = ("easy", "medium", "hard")

# 0.0 -> spend ~ balance/k on every target.
# 1.0 -> allocate budget purely in proportion to each target's marginal value
CONCENTRATION = 0.8

# Handles near-ties / equivalent substitutes.
SUBSTITUTE_TOLERANCE = 0.1

# Only run the combinatorial search over this many candidates.
MAX_CANDIDATES = 15


def _team_scorer(missing_position_penalty: int):
    penalty_factor = 1.0 / (2.0 ** (missing_position_penalty / 4.0))

    def get_team_score(team: list[NBAPlayer]) -> float:
        temp_player = Player(name="temp")
        temp_player.nba_team = team
        shortfall = temp_player.best_lineup()
        return temp_player.compute_score(shortfall, penalty_factor)

    return get_team_score


def compute_bid_easy(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue or balance <= 0:
        return 0
    if len(player_queue) <= k:
        return 1

    current_player = player_queue[0]

    penalty_factor = 1.0 / (2.0 ** (missing_position_penalty / 4.0))

    # Reuse a single Player instance as a scorer to avoid re-validating pydantic
    # models on every one of the (potentially tens of thousands of) evaluations.
    scorer = Player(name="__scorer__")

    def team_score(extra: list[NBAPlayer]) -> float:
        scorer.nba_team = current_team + extra
        return scorer.compute_score(scorer.best_lineup(), penalty_factor)

    def solo_value(p: NBAPlayer) -> float:
        scorer.nba_team = [p]
        return scorer.compute_score(scorer.best_lineup(), penalty_factor)

    def best_team(pool: list[NBAPlayer], size: int) -> tuple[float, list[NBAPlayer]]:
        if size <= 0 or len(pool) < size:
            return -1.0, []
        best_score = -1.0
        best_combo: tuple[NBAPlayer, ...] | None = None
        for combo in itertools.combinations(pool, size):
            s = team_score(list(combo))
            if s > best_score:
                best_score = s
                best_combo = combo
        return best_score, list(best_combo) if best_combo else []

    def best_team_including(player: NBAPlayer, pool: list[NBAPlayer], size: int) -> float:
        others = [p for p in pool if p.pid != player.pid]
        if size - 1 > len(others):
            return -1.0
        best_score = -1.0
        for combo in itertools.combinations(others, size - 1):
            s = team_score([player] + list(combo))
            if s > best_score:
                best_score = s
        return best_score

    # 1. Prune to the strongest candidates, then guarantee we can still field a
    #    legal lineup (centers in particular can be scarce) and that the player
    #    on the block is always evaluated.
    ranked = sorted(player_queue, key=solo_value, reverse=True)
    candidates = ranked[:MAX_CANDIDATES]
    cand_ids = {p.pid for p in candidates}
    if current_player.pid not in cand_ids:
        candidates.append(current_player)
        cand_ids.add(current_player.pid)
    for pos, need in _REQUIRED_POS_COUNTS.items():
        have = sum(1 for c in candidates if getattr(c, pos))
        for p in ranked:
            if have >= need:
                break
            if p.pid not in cand_ids and getattr(p, pos):
                candidates.append(p)
                cand_ids.add(p.pid)
                have += 1

    # 2. Best team now, and best team if we lose the current player.
    u_best, target_team = best_team(candidates, k)
    without_current = [p for p in candidates if p.pid != current_player.pid]
    u_without_current, _ = best_team(without_current, k)

    target_ids = {p.pid for p in target_team}
    current_is_target = current_player.pid in target_ids

    # 3. If the current player isn't in our optimal team, only chase him if he's
    #    a near-tie substitute; otherwise wait for the players we actually want.
    if not current_is_target:
        u_with_current = best_team_including(current_player, candidates, k)
        is_substitute = u_best > 0 and u_with_current >= u_best * (1.0 - SUBSTITUTE_TOLERANCE)
        if not is_substitute:
            return 0
        current_is_target = True

    # 4. Marginal (irreplaceability) value of each planned player: how much the
    #    optimal team score drops if that player is removed from the pool.
    marginal = {}
    for m in target_team:
        reduced = [p for p in candidates if p.pid != m.pid]
        s_wo, _ = best_team(reduced, k)
        marginal[m.pid] = max(0.0, u_best - s_wo)

    # Assemble the k slots we plan to buy. If the current player is a substitute
    # (not literally in target_team), he competes for the most replaceable slot.
    alloc = list(target_team)
    mv_current = max(0.0, u_best - u_without_current)
    if current_player.pid not in target_ids:
        weakest = min(target_team, key=lambda p: marginal[p.pid])
        alloc = [current_player if p.pid == weakest.pid else p for p in target_team]
    marginal[current_player.pid] = mv_current

    # 5. Allocate the budget: blend an even split (guarantees we keep bidding to
    #    fill the roster) with a marginal-value-proportional split (concentrates
    #    spend on irreplaceable talent).
    total_mv = sum(marginal[p.pid] for p in alloc)

    def share_of(p: NBAPlayer) -> float:
        if total_mv <= 0:
            return 1.0 / k
        return (1.0 - CONCENTRATION) / k + CONCENTRATION * (marginal[p.pid] / total_mv)

    bid = balance * share_of(current_player)

    # Keep at least $1 per remaining slot so we can always stay in future
    # auctions; the free-assignment mechanic covers the true worst case.
    reserve = max(0, min(k - 1, balance - 1))
    bid = min(bid, balance - reserve)
    bid = int(round(bid))

    # We want this player, so never let a $0 bid beat us for him.
    bid = max(1, min(bid, balance))
    return bid


def compute_bid_medium(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    k = 5 - len(current_team)
    if k <= 0:
        return 0

    if not player_queue:
        return 0

    if len(player_queue) <= k:
        return 1

    current_player = player_queue[0]
    get_team_score = _team_scorer(missing_position_penalty)

    # 1. For each candidate player P, compute their utility U(P).
    # U(P) is the max score we can get if we draft P next and fill the remaining
    # k-1 slots optimally from the rest of the candidates.
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

    # 2. Sort candidates by utility descending
    sorted_candidates = sorted(player_queue, key=lambda x: utilities[x.pid], reverse=True)

    # Our target players are the top k players
    target_players = sorted_candidates[:k]
    target_pids = {p.pid for p in target_players}

    # 3. Determine the replacement utility level.
    # The replacement player is the average of the players we don't target.
    end_index = -additional_players if additional_players else None
    replacement_players = sorted_candidates[k:end_index]
    if len(replacement_players):
        u_replacement = sum(utilities[p.pid] for p in replacement_players) / len(replacement_players)
    else:
        u_replacement = 0

    # 4. Compute each player's value.
    #
    # For NON-target players, value over replacement (utility minus the
    # replacement baseline) tells us how good they are relative to a generic
    # fallback -- useful for ranking non-targets against each other for a
    # small "insurance" bid (denying them to an opponent / bench depth in
    # case we miss our real targets). But it is NOT on the same scale as the
    # marginal contribution values above, and comparing them directly caused
    # a real bug: it's possible for a decent non-target's raw VORP number to
    # exceed a legitimate target's marginal value (this happens whenever a
    # target is only barely better than the best available fallback, i.e.
    # positionally "covered" by a teammate already in the target group, e.g.
    # a second-string center once you already have an elite one). Left
    # unscaled, that let the bot bid MORE on a player it doesn't even want
    # than on one of its actual targets.
    #
    # Fix: convert non-target VORP into the same units as marginal value by
    # scaling it relative to our *weakest* target's own VORP-over-replacement
    # figure, using our weakest target's marginal value as the conversion
    # rate. Since every non-target's VORP is, by construction, <= the
    # weakest target's VORP (they didn't make the top-k cut), this
    # guarantees every non-target's scaled value is capped at the weakest
    # target's marginal value -- an insurance bid can never outbid an actual
    # target, while still preserving relative ranking among non-targets.
    #
    # For TARGET players, utility itself is NOT a usable measure of value
    # either (that was the earlier bug): U(P) for any P in the optimal
    # top-k group reconstructs the same optimal lineup regardless of which
    # target you start from, so it comes out identical for all of them. We
    # instead value each target by its MARGINAL contribution -- the drop in
    # our best achievable score if that specific player were unavailable
    # and we had to fill the slot with our best remaining alternative. A
    # true stand-out shows a huge drop; a "good but substitutable" target
    # (e.g. a second center once you already have a great one) shows almost
    # none -- which is exactly the case that originally exposed this bug.
    best_team_score = get_team_score(current_team + target_players)

    marginal_values = {}
    for P in target_players:
        others = [x for x in player_queue if x.pid != P.pid]
        if len(others) >= k:
            best_without_P = max(
                get_team_score(current_team + list(subset))
                for subset in itertools.combinations(others, k)
            )
        else:
            # Not enough remaining candidates to complete a roster without P
            # -- treat P as effectively irreplaceable.
            best_without_P = get_team_score(current_team)
        marginal_values[P.pid] = max(0.0, best_team_score - best_without_P)

    weakest_target = min(target_players, key=lambda p: marginal_values[p.pid])
    weakest_target_marginal = marginal_values[weakest_target.pid]
    weakest_target_vorp = max(0.0, utilities[weakest_target.pid] - u_replacement)

    def non_target_value(p) -> float:
        if weakest_target_vorp <= 0:
            return 0.0
        vorp = max(0.0, utilities[p.pid] - u_replacement)
        return weakest_target_marginal * (vorp / weakest_target_vorp)

    values = {
        p.pid: (marginal_values[p.pid] if p.pid in target_pids
                else non_target_value(p))
        for p in player_queue
    }

    # If the current player has no value over replacement, we bid 0
    current_player_value = values[current_player.pid]
    if current_player_value <= 0:
        return 0

    # 5. Allocate budget proportional to value over replacement
    sum_target_v = sum(values[p.pid] for p in target_players)

    if sum_target_v == 0:
        # If all target players have 0 value over replacement (e.g., all identical),
        # we distribute budget equally among target players
        is_target = any(p.pid == current_player.pid for p in target_players)
        if is_target:
            bid = max(1, balance // k)
        else:
            bid = 0
    else:
        # Bid is proportional to current player's value vs the sum of target values
        bid = balance * (current_player_value / sum_target_v)
        bid = int(round(bid))

    # Smart micro-optimizations:
    # 1. If we want this player (value > 0) and have budget, bid at least $1 so we don't lose to a $0 bid.
    if bid == 0 and balance > 0 and current_player_value > 0:
        bid = 1

    # 2. Ensure the bid never exceeds our current balance or goes below 0.
    bid = max(0, min(bid, balance))

    return bid


def compute_bid_hard(
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    additional_players: int,
    balance: int,
    current_team: list[NBAPlayer],
) -> int:
    pass


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
