import itertools
import math
from game import NBAPlayer
from player import Player, _REQUIRED_POS_COUNTS
from pydantic import BaseModel

DIFFICULTIES = ("easy", "medium", "hard")
SUBSTITUTION_THRESHOLD = 0.84

# Candidate pool size for exhaustive best-team search. Kept constant
# regardless of room size so bid computation stays fast.
_CANDIDATE_POOL_SIZE = 15

# --- Medium bot tunables ---
# In a second-price auction the winner pays the runner-up's bid, so bidding
# above the "fair" proportional share is nearly free upside: it converts
# close losses into wins without raising the price we actually pay.
_MEDIUM_AGGRESSION = 1.5
# Snipe (bid 1) on a non-target player when forcing him onto the roster still
# achieves at least this fraction of the unconstrained best team score.
# Opponents bidding 0 hand us the player at price 0.
_MEDIUM_SNIPE_RATIO = 0.97
# Looser snipe ratio for players about to be force-assigned to a random
# roster: claiming them deterministically beats the dump lottery.
_MEDIUM_DUMP_SNIPE_RATIO = 0.92
# Bid everything (minus slot reserve) on the highest-marginal remaining
# target: no better opportunity exists later in the queue.
_MEDIUM_ALL_IN_TOP = True

# --- Hard bot tunables ---
# The hard bot evaluates a pessimistic free completion as well as the best
# completion. Searching a small, team-specific low-end pool keeps that lower
# bound inexpensive even in large rooms.
_HARD_FLOOR_POOL_SIZE = 13
# Turn relative budget-per-open-slot into a position between the pessimistic
# free team and the best team. An exponent below one reflects that the floor
# is deliberately adversarial rather than an average free roster.
_HARD_MARKET_CURVE = 0.35
_HARD_MARGIN_TOLERANCE = 0.002
_HARD_SNIPE_RATIO = 0.985
_HARD_DUMP_SNIPE_RATIO = 0.94
_SCORE_EPSILON = 1e-12


def _fast_team_scorer(missing_position_penalty: int, base_stats: list[tuple]):
    """Returns eval(extra_stats) mirroring Player.compute_score + best_lineup.

    Operates on plain tuples with a memoized position-matching shortfall so it
    is fast enough for exhaustive combination search. Base roster aggregates
    are precomputed once; each eval only folds in the candidate extras.
    """
    penalty = 1.0 / (2.0 ** (missing_position_penalty / 4.0))
    shortfall_memo: dict[tuple[int, ...], int] = {}
    caps = (
        _REQUIRED_POS_COUNTS['guard'],
        _REQUIRED_POS_COUNTS['forward'],
        _REQUIRED_POS_COUNTS['center'],
    )
    total_required = sum(caps)

    base_pts = base_ast = base_reb = base_blk = base_stl = 0.0
    base_tov = base_tsm = base_tsa = 0.0
    base_masks: list[int] = []
    for s in base_stats:
        base_pts += s[0]; base_ast += s[1]; base_reb += s[2]; base_blk += s[3]
        base_stl += s[4]; base_tov += s[5]
        base_tsm += s[6] * s[7]; base_tsa += s[7]
        base_masks.append(s[8])

    def shortfall(masks: tuple[int, ...]) -> int:
        key = tuple(sorted(masks))
        cached = shortfall_memo.get(key)
        if cached is not None:
            return cached
        # Exact max bipartite matching via DP over remaining slot capacities.
        states = {(0, 0, 0): 0}
        for mask in key:
            nxt = dict(states)
            for (g, f, c), filled in states.items():
                if mask & 1 and g < caps[0]:
                    s = (g + 1, f, c)
                    if nxt.get(s, -1) < filled + 1:
                        nxt[s] = filled + 1
                if mask & 2 and f < caps[1]:
                    s = (g, f + 1, c)
                    if nxt.get(s, -1) < filled + 1:
                        nxt[s] = filled + 1
                if mask & 4 and c < caps[2]:
                    s = (g, f, c + 1)
                    if nxt.get(s, -1) < filled + 1:
                        nxt[s] = filled + 1
            states = nxt
        result = total_required - max(states.values())
        shortfall_memo[key] = result
        return result

    def eval_extras(extras: list[tuple]) -> float:
        pts, ast, reb, blk, stl = base_pts, base_ast, base_reb, base_blk, base_stl
        tov, tsm, tsa = base_tov, base_tsm, base_tsa
        masks = list(base_masks)
        for s in extras:
            pts += s[0]; ast += s[1]; reb += s[2]; blk += s[3]
            stl += s[4]; tov += s[5]
            tsm += s[6] * s[7]; tsa += s[7]
            masks.append(s[8])
        if tov <= 0:
            tov = 1e-9
        if tsa <= 0:
            tsa = 1e-9
        ts = tsm / tsa
        base = (pts ** 1.2 * ast * reb * blk ** 0.2 * stl ** 0.2
                * (blk + stl) ** 0.4 * ts ** 1.5 / math.sqrt(tov))
        return base * penalty ** shortfall(tuple(masks))

    return eval_extras


def _stat_tuple(p: NBAPlayer) -> tuple:
    mask = (1 if p.guard else 0) | (2 if p.forward else 0) | (4 if p.center else 0)
    return (p.pts, p.ast, p.reb, p.blk, p.stl, p.tov, p.ts, p.tsa, mask)


def _rank_score(p: NBAPlayer) -> float:
    """Solo strength used only for ranking candidates. Smoothed so players
    with a zero stat (e.g. 0.0 blocks) don't all collapse to rank 0."""
    return (max(p.pts, 0.1) ** 1.2 * max(p.ast, 0.1) * max(p.reb, 0.1)
            * max(p.blk, 0.05) ** 0.2 * max(p.stl, 0.05) ** 0.2
            * (p.blk + p.stl + 0.05) ** 0.4
            * max(p.ts, 0.1) ** 1.5 / math.sqrt(max(p.tov, 0.25)))


def _build_candidate_pool(
    ranked: list[NBAPlayer],
    current_team: list[NBAPlayer],
    ensure_include: NBAPlayer | None = None,
) -> list[NBAPlayer]:
    """Top-_CANDIDATE_POOL_SIZE players by rank, plus position coverage fills."""
    pool = ranked[:_CANDIDATE_POOL_SIZE]
    pool_ids = {p.pid for p in pool}
    if ensure_include is not None and ensure_include.pid not in pool_ids:
        pool.append(ensure_include)
        pool_ids.add(ensure_include.pid)
    for pos, need in _REQUIRED_POS_COUNTS.items():
        have = (sum(1 for p in pool if getattr(p, pos))
                + sum(1 for p in current_team if getattr(p, pos)))
        for cand in ranked:
            if have >= need:
                break
            if cand.pid not in pool_ids and getattr(cand, pos):
                pool.append(cand)
                pool_ids.add(cand.pid)
                have += 1
    return pool


def _make_search(
    missing_position_penalty: int,
    current_team: list[NBAPlayer],
    cand_stats: list[tuple]
):
    """Build memoized eval_team / best_team over candidate indices."""
    eval_extras = _fast_team_scorer(
        missing_position_penalty, [_stat_tuple(p) for p in current_team]
    )
    eval_memo: dict[frozenset[int], float] = {}

    def eval_team(extra_indices: tuple[int]) -> float:
        key = frozenset(extra_indices)
        cached = eval_memo.get(key)
        if cached is not None:
            return cached
        score = eval_extras([cand_stats[i] for i in extra_indices])
        eval_memo[key] = score
        return score

    def best_team(indices: tuple[int], choose: int, forced: tuple[int] = ()) -> tuple[float, tuple[int, ...]]:
        best_score, best_combo = 0, ()
        for combo in itertools.combinations(indices, choose):
            s = eval_team(forced + combo)
            if s > best_score:
                best_score, best_combo = s, combo
        return best_score, best_combo

    return best_team


def compute_bid_easy(
    missing_position_penalty: int,
    additional_players: int,
    room_players: list[Player],
    player_queue: list[NBAPlayer],
    current_team: list[NBAPlayer],
    balance: int,
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue or balance <= 0:
        return 0
    # We are forced to take all remaining players in the queue.
    if len(player_queue) <= k:
        return 1

    current = player_queue[0]
    ranked = sorted(player_queue, key=_rank_score, reverse=True)
    pool = _build_candidate_pool(ranked, current_team, ensure_include=current)

    cand_stats = [_stat_tuple(p) for p in pool]
    all_indices = tuple(range(len(cand_stats)))
    current_idx = next(i for i, p in enumerate(pool) if p.pid == current.pid)
    best_team = _make_search(missing_position_penalty, current_team, cand_stats)

    u_best, target = best_team(all_indices, k)
    target_set = set(target)

    marginals = {}
    for t in target:
        others = tuple(i for i in all_indices if i != t)
        marginals[t] = u_best - best_team(others, k)[0]
    marginals_total = sum(marginals.values())

    if current_idx in target_set:
        if not marginals_total:
            share = 1 / k
        else:
            share = marginals[current_idx] / marginals_total
        return max(1, int(balance * share))

    # Score current first so we can bail before evaluating every non-target.
    others = tuple(i for i in all_indices if i != current_idx)
    u_current = best_team(others, k - 1, forced=(current_idx,))[0]
    if u_current < SUBSTITUTION_THRESHOLD * u_best:
        return 0

    non_target_utilities = {current_idx: u_current}
    for i in all_indices:
        if i in target_set or i == current_idx:
            continue
        others = tuple(j for j in all_indices if j != i)
        non_target_utilities[i] = best_team(others, k - 1, forced=(i,))[0]

    end_index = -additional_players if additional_players else None
    replacement_utilities = sorted(non_target_utilities.values(), reverse=True)[:end_index]
    u_avg = sum(replacement_utilities) / len(replacement_utilities) if replacement_utilities else 0

    if u_current < u_avg:
        return 0

    def vorp(u: float) -> float:
        return max(0.0, u - u_avg)

    vorp_total = sum(vorp(u) for u in replacement_utilities)
    if not vorp_total or not marginals_total:
        return 0
    vorp_share = vorp(u_current) / vorp_total
    share = vorp_share * min(marginals.values()) / marginals_total
    return max(1, int(balance * share))


def compute_bid_medium(
    missing_position_penalty: int,
    additional_players: int,
    room_players: list[Player],
    player_queue: list[NBAPlayer],
    current_team: list[NBAPlayer],
    balance: int,
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue or balance <= 0:
        return 0
    # We need every remaining player. Easy bids 1 here; 2 outbids it while the
    # second-price rule keeps our cost at most 1.
    if len(player_queue) <= k:
        return 2

    current = player_queue[0]
    future = player_queue[1:]

    # Identify ourselves among the room players. _submit_bot_bids passes the
    # member's own nba_team list, so an identity check works; fall back to
    # matching state in case a caller passes copies.
    me = next((p for p in room_players if p.nba_team is current_team), None)
    if me is None:
        my_pids = sorted(x.pid for x in current_team)
        me = next(
            (p for p in room_players
             if p.balance == balance and sorted(x.pid for x in p.nba_team) == my_pids),
            None,
        )
    opponents = [p for p in room_players if p is not me]
    max_opp_balance = max(
        (o.balance for o in opponents if len(o.nba_team) < 5), default=0
    )

    ranked_future = sorted(future, key=_rank_score, reverse=True)
    pool = _build_candidate_pool(ranked_future, current_team)

    cand_stats = [_stat_tuple(current)] + [_stat_tuple(p) for p in pool]
    all_indices = tuple(range(len(cand_stats)))
    CURRENT = 0
    best_team = _make_search(missing_position_penalty, current_team, cand_stats)

    u_best, target = best_team(all_indices, k)
    without_current = tuple(i for i in all_indices if i != CURRENT)
    bid_cap = max(1, balance - (k - 1))  # keep >= 1 dollar per remaining slot

    if CURRENT in target:
        marginals = {}
        for t in target:
            others = tuple(i for i in all_indices if i != t)
            marginals[t] = max(0.0, u_best - best_team(others, k)[0])
        total = sum(marginals.values())
        share = marginals[CURRENT] / total if total > 0 else 1.0 / k
        if _MEDIUM_ALL_IN_TOP and marginals[CURRENT] >= max(marginals.values()):
            # No better opportunity remains in the queue; the second-price
            # rule refunds whatever the runner-up didn't force us to pay.
            bid = bid_cap
        else:
            bid = max(1, min(int(balance * share * _MEDIUM_AGGRESSION), bid_cap))
        # Bidding above the richest opponent's balance can't change anything.
        return min(bid, max_opp_balance + 1)

    # Not in the target team: snipe near-equivalent players with a minimal
    # bid. Opponents that bid 0 hand them to us at price 0, filling a roster
    # slot nearly optimally while preserving budget for contested targets.
    u_forced, _ = best_team(without_current, k - 1, forced=(CURRENT,))
    snipe_ratio = _MEDIUM_SNIPE_RATIO
    if current.skipped >= additional_players:
        snipe_ratio = _MEDIUM_DUMP_SNIPE_RATIO
    if u_best > 0 and u_forced >= snipe_ratio * u_best:
        return 1
    return 0


def _hard_best_completion(
    missing_position_penalty: int,
    current_team: list[NBAPlayer],
    available: list[NBAPlayer],
    choose: int,
) -> float:
    """Best completion score from a small high-upside candidate pool."""
    if choose <= 0:
        return _fast_team_scorer(
            missing_position_penalty, [_stat_tuple(p) for p in current_team]
        )([])
    if len(available) < choose:
        return 0.0

    ranked = sorted(available, key=_rank_score, reverse=True)
    pool = _build_candidate_pool(ranked, current_team)
    if len(pool) < choose:
        pool = ranked
    cand_stats = [_stat_tuple(p) for p in pool]
    best_team = _make_search(
        missing_position_penalty, current_team, cand_stats
    )
    return best_team(tuple(range(len(pool))), choose)[0]


def _hard_free_completion(
    missing_position_penalty: int,
    current_team: list[NBAPlayer],
    available: list[NBAPlayer],
    choose: int,
    disposable_players: int,
) -> float:
    """Pessimistic score obtainable without winning another auction.

    The worst ``disposable_players`` are first removed: those players can be
    left unassigned at the end of the game. Among the players that must be
    assigned, search a low-end pool for the worst completion. This implements
    a useful absolute baseline without exhaustively searching a large queue.
    """
    eval_extras = _fast_team_scorer(
        missing_position_penalty, [_stat_tuple(p) for p in current_team]
    )
    if choose <= 0:
        return eval_extras([])
    if len(available) < choose:
        return 0.0

    ranked_low = sorted(available, key=_rank_score)
    can_discard = max(0, len(ranked_low) - choose)
    discard = min(max(0, disposable_players), can_discard)
    survivors = ranked_low[discard:]

    # Once truly disposable players are gone, rank the low end in the context
    # of this roster. This catches position and stat-complement edge cases that
    # a player-only ranking can miss.
    survivor_stats = [(_stat_tuple(p), p) for p in survivors]
    survivor_stats.sort(key=lambda item: eval_extras([item[0]]))
    pool = survivor_stats[:max(choose, min(_HARD_FLOOR_POOL_SIZE, len(survivors)))]

    worst = math.inf
    for combo in itertools.combinations(pool, choose):
        score = eval_extras([item[0] for item in combo])
        if score < worst:
            worst = score
    return worst if worst < math.inf else 0.0


def _hard_completion_bounds(
    missing_position_penalty: int,
    current_team: list[NBAPlayer],
    available: list[NBAPlayer],
    choose: int,
    disposable_players: int,
) -> tuple[float, float]:
    """Return (free-floor score, best score) for a partial roster."""
    ceiling = _hard_best_completion(
        missing_position_penalty, current_team, available, choose
    )
    floor = _hard_free_completion(
        missing_position_penalty,
        current_team,
        available,
        choose,
        disposable_players,
    )
    if ceiling <= 0:
        return floor, floor
    return min(floor, ceiling), ceiling


def _hard_project_scores(
    bounds: list[tuple[float, float]],
    balances: list[int],
    open_slots: list[int],
    market_curve: float,
) -> list[float]:
    """Project final scores from completion bounds and market buying power."""
    rates = [
        balance / slots if balance > 0 and slots > 0 else 0.0
        for balance, slots in zip(balances, open_slots)
    ]
    projected: list[float] = []
    for i, ((floor, ceiling), slots) in enumerate(zip(bounds, open_slots)):
        floor = max(floor, _SCORE_EPSILON)
        ceiling = max(ceiling, floor)
        if slots <= 0 or ceiling <= floor:
            projected.append(ceiling)
            continue

        own_rate = rates[i]
        competition = math.sqrt(sum(
            rate * rate for j, rate in enumerate(rates) if j != i
        ))
        if own_rate <= 0:
            quality = 0.0
        elif competition <= 0:
            quality = 1.0
        else:
            market_share = own_rate / (own_rate + competition)
            quality = market_share ** market_curve
        projected.append(
            math.exp(math.log(floor) + quality * math.log(ceiling / floor))
        )
    return projected


def _hard_margin(scores: list[float], player_idx: int) -> float:
    """Log-score lead over the strongest opponent."""
    own = math.log(max(scores[player_idx], _SCORE_EPSILON))
    rivals = [
        math.log(max(score, _SCORE_EPSILON))
        for i, score in enumerate(scores)
        if i != player_idx
    ]
    return own - max(rivals) if rivals else own


def _hard_find_self(
    room_players: list[Player],
    current_team: list[NBAPlayer],
    balance: int,
) -> int:
    """Find this hard bot's Player entry despite BotInputs copying the list."""
    pids = sorted(p.pid for p in current_team)
    state_matches = [
        i for i, player in enumerate(room_players)
        if player.balance == balance
        and sorted(p.pid for p in player.nba_team) == pids
    ]
    for i in state_matches:
        if room_players[i].bot_difficulty == "hard":
            return i
    for i, player in enumerate(room_players):
        if player.bot_difficulty == "hard":
            return i
    return state_matches[0] if state_matches else 0


def compute_bid_hard(
    missing_position_penalty: int,
    additional_players: int,
    room_players: list[Player],
    player_queue: list[NBAPlayer],
    current_team: list[NBAPlayer],
    balance: int,
) -> int:
    slots_needed = 5 - len(current_team)
    if slots_needed <= 0 or not player_queue or balance <= 0:
        return 0

    current = player_queue[0]
    future = player_queue[1:]
    open_slots = [max(0, 5 - len(player.nba_team)) for player in room_players]
    total_open_slots = sum(open_slots)

    # The queue/open-slot difference is the number that may finish unassigned.
    # Prefer live state over the initial setting so this also handles direct
    # callers and unusual test fixtures correctly.
    disposable = max(0, len(player_queue) - total_open_slots)
    if total_open_slots <= 0:
        return 0
    # More surplus players make a high-quality completion attainable without
    # dominating the budget market. Move projections toward the ceiling as
    # supply slack grows, while retaining the adversarial free-team floor.
    market_curve = (
        _HARD_MARKET_CURVE
        * total_open_slots
        / max(1, total_open_slots + disposable)
    )

    me_idx = _hard_find_self(room_players, current_team, balance)
    balances = [player.balance for player in room_players]
    balances[me_idx] = balance
    open_slots[me_idx] = slots_needed

    # Only positive-balance incomplete opponents can set a second price.
    active_opponents = [
        i for i, (b, slots) in enumerate(zip(balances, open_slots))
        if i != me_idx and b > 0 and slots > 0
    ]
    if len(future) < slots_needed:
        return 1

    without_bounds: list[tuple[float, float]] = []
    with_bounds: list[tuple[float, float]] = []
    for i, player in enumerate(room_players):
        team = current_team if i == me_idx else player.nba_team
        slots = open_slots[i]
        without_bounds.append(_hard_completion_bounds(
            missing_position_penalty, team, future, slots, disposable
        ))
        if slots > 0:
            with_bounds.append(_hard_completion_bounds(
                missing_position_penalty,
                team + [current],
                future,
                slots - 1,
                disposable,
            ))
        else:
            with_bounds.append(without_bounds[-1])

    base_scores = _hard_project_scores(
        without_bounds, balances, open_slots, market_curve
    )

    # Estimate each rival's reservation price from the same roster/budget
    # fundamentals used for ourselves. This is deliberately strategy-agnostic:
    # it works for humans and future bots, not only the current easy/medium
    # implementations.
    opponent_reservations: dict[int, int] = {}
    opponent_zero_scores: dict[int, float] = {}
    for i in active_opponents:
        reservation = 0
        score_at_zero = 0.0
        for price in range(balances[i] + 1):
            trial_bounds = list(without_bounds)
            trial_bounds[i] = with_bounds[i]
            trial_balances = list(balances)
            trial_balances[i] -= price
            trial_slots = list(open_slots)
            trial_slots[i] -= 1
            score = _hard_project_scores(
                trial_bounds, trial_balances, trial_slots, market_curve
            )[i]
            if price == 0:
                score_at_zero = score
            if score + _SCORE_EPSILON >= base_scores[i]:
                reservation = price
        if (
            reservation == 0
            and score_at_zero >= _HARD_SNIPE_RATIO * base_scores[i]
        ):
            reservation = 1
        opponent_reservations[i] = reservation
        opponent_zero_scores[i] = score_at_zero

    likely_winner = None
    if opponent_reservations:
        likely_winner = max(
            opponent_reservations,
            key=lambda i: (
                opponent_reservations[i],
                opponent_zero_scores[i] / max(base_scores[i], _SCORE_EPSILON),
                balances[i],
            ),
        )
        if opponent_reservations[likely_winner] <= 0:
            likely_winner = None

    def scenario_scores(recipient: int, price_paid: int) -> list[float]:
        scenario_bounds = list(without_bounds)
        scenario_bounds[recipient] = with_bounds[recipient]
        scenario_balances = list(balances)
        scenario_balances[recipient] = max(
            0, scenario_balances[recipient] - price_paid
        )
        scenario_slots = list(open_slots)
        scenario_slots[recipient] = max(0, scenario_slots[recipient] - 1)
        return _hard_project_scores(
            scenario_bounds, scenario_balances, scenario_slots, market_curve
        )

    reservation = 0
    if likely_winner is None:
        # With no expected competing bid, money is not the opportunity cost;
        # compare taking the player for free with preserving the future queue.
        for price in range(balance + 1):
            if (
                scenario_scores(me_idx, price)[me_idx] + _SCORE_EPSILON
                >= base_scores[me_idx]
            ):
                reservation = price
    else:
        # A losing bid becomes the second price. Compare the championship
        # margin when we win and pay that price with the margin when the most
        # interested rival wins and is made to pay it. This includes both
        # player value and denial value, while naturally preserving budget for
        # stronger players later in the known queue.
        for price in range(balance + 1):
            win_scores = scenario_scores(me_idx, price)
            rival_price = min(price, balances[likely_winner])
            lose_scores = scenario_scores(likely_winner, rival_price)
            if (
                _hard_margin(win_scores, me_idx) + _HARD_MARGIN_TOLERANCE
                >= _hard_margin(lose_scores, me_idx)
            ):
                reservation = price

    if reservation <= 0:
        # On the force-assignment pass, claim a near-equivalent player for one
        # rather than accepting a random recipient. The strict ratio prevents
        # a bad player from consuming a valuable roster slot.
        force_assignment_imminent = current.skipped > max(
            disposable, additional_players
        )
        take_for_free = scenario_scores(me_idx, 0)[me_idx]
        ratio = (
            _HARD_DUMP_SNIPE_RATIO
            if force_assignment_imminent
            else _HARD_SNIPE_RATIO
        )
        if take_for_free >= ratio * base_scores[me_idx]:
            return 1
        return 0

    max_opponent_balance = max(
        (balances[i] for i in active_opponents), default=0
    )
    if max_opponent_balance <= 0:
        return 1
    # More than one dollar above the richest possible rival cannot change the
    # auction result, and a positive reservation should always submit a
    # positive integer bid.
    return max(1, min(reservation, balance, max_opponent_balance + 1))


_STRATEGIES = {
    "easy": compute_bid_easy,
    "medium": compute_bid_medium,
    "hard": compute_bid_hard,
}

class BotInputs(BaseModel):
    # Room settings
    missing_position_penalty: int
    additional_players: int

    # Room state
    room_players: list[Player]
    player_queue: list[NBAPlayer]

    # Bot state
    current_team: list[NBAPlayer]
    balance: int

def compute_bid(difficulty: str, inputs: BotInputs) -> int:
    strategy = _STRATEGIES[difficulty]
    return strategy(**{field: getattr(inputs, field) for field in BotInputs.model_fields})
