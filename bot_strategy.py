import itertools
import math
from game import NBAPlayer
from player import Player, _REQUIRED_POS_COUNTS
from pydantic import BaseModel

DIFFICULTIES = ("easy", "medium", "hard")
SUBSTITUTION_THRESHOLD = 0.84
MAX_CANDIDATES_RATIO = 5

def _team_scorer(missing_position_penalty: int):
    penalty_factor = 1.0 / (2.0 ** (missing_position_penalty / 4.0))
    scorer = Player(name="__scorer__")

    def get_team_score(team: list[NBAPlayer]) -> float:
        scorer.nba_team = team
        return scorer.compute_score(scorer.best_lineup(), penalty_factor)

    return get_team_score

def compute_bid_easy(
    missing_position_penalty: int,
    additional_players: int,
    room_players: list[Player],
    player_queue: list[NBAPlayer],
    current_team: list[NBAPlayer],
    balance: int,
    **kwargs
) -> int:
    k = 5 - len(current_team)
    if k <= 0 or not player_queue or balance <= 0:
        return 0
    # We are forced to take all remaining players in the queue.
    if len(player_queue) <= k:
        return 1

    get_team_score = _team_scorer(missing_position_penalty)

    def team_score(extra: list[NBAPlayer]) -> float:
        return get_team_score(current_team + extra)

    def solo_score(player: NBAPlayer) -> float:
        return get_team_score([player])

    def pool_minus(pool: list[NBAPlayer], minus: NBAPlayer) -> list[NBAPlayer]:
        return [p for p in pool if p.pid != minus.pid]

    def best_team(pool: list[NBAPlayer], spots_remaining: int) -> tuple[float, list[NBAPlayer]]:
        best_score = 0
        best_combo: tuple[NBAPlayer] = ()
        for combo in itertools.combinations(pool, spots_remaining):
            s = team_score(list(combo))
            if s > best_score:
                best_score = s
                best_combo = combo
        return best_score, list(best_combo)

    def best_team_including(pool: list[NBAPlayer], spots_remaining: int, including: NBAPlayer) -> float:
        cands = pool_minus(pool, including)
        return max(team_score(list(combo) + [including]) for combo in itertools.combinations(cands, spots_remaining - 1))

    current_player = player_queue[0]

    # Prune player_queue to strongest candidates, ensure current player
    # is included and required position counts are met.
    ranked = sorted(player_queue, key=solo_score, reverse=True)
    pool = ranked[:MAX_CANDIDATES_RATIO * len(room_players)]
    pool_ids = {p.pid for p in pool}
    if current_player.pid not in pool_ids:
        pool.append(current_player)
        pool_ids.add(current_player.pid)
    for pos, need in _REQUIRED_POS_COUNTS.items():
        have = sum(1 for p in pool if getattr(p, pos))
        for player in ranked:
            if have >= need:
                break
            if player.pid not in pool_ids and getattr(player, pos):
                pool.append(player)
                pool_ids.add(player.pid)
                have += 1

    u_best, target_team = best_team(pool, k)
    target_ids = {p.pid for p in target_team}
    marginals = {
        t.pid: u_best - best_team(pool_minus(pool, t), k)[0]
        for t in target_team
    }
    marginals_total = sum(marginals.values())
    if current_player.pid in target_ids:
        if not marginals_total:
            share = 1 / k
        else:
            share = marginals[current_player.pid] / marginals_total
        return max(1, int(balance * share))

    non_target_utilities = {
        p.pid: best_team_including(pool, k, p) for p in pool if p.pid not in target_ids
    }
    u_current = non_target_utilities[current_player.pid]
    end_index = -additional_players if additional_players else None
    replacement_utilities = sorted(non_target_utilities.values(), reverse=True)[:end_index]
    u_avg = sum(replacement_utilities) / len(replacement_utilities) if replacement_utilities else 0

    if u_current < u_avg or u_current < SUBSTITUTION_THRESHOLD * u_best:
        return 0

    def vorp(u: float, avg: float) -> float:
        return max(0, u - avg)
    
    vorp_total = sum(vorp(u, u_avg) for u in replacement_utilities)
    if not vorp_total:
        return 0
    vorp_share = vorp(u_current, u_avg) / vorp_total
    share = vorp_share * min(marginals.values()) / marginals_total
    return max(1, int(balance * share))

# --- Medium bot tunables ---
# Candidate pool size for the exhaustive best-team search. Kept constant
# regardless of room size so bid computation stays fast.
_MEDIUM_POOL_SIZE = 15
# In a second-price auction the winner pays the runner-up's bid, so bidding
# above the "fair" proportional share is nearly free upside: it converts
# close losses into wins without raising the price we actually pay.
_MEDIUM_AGGRESSION = 2.5
# Snipe (bid 1) on a non-target player when forcing him onto the roster still
# achieves at least this fraction of the unconstrained best team score.
# Opponents bidding 0 hand us the player at price 0.
_MEDIUM_SNIPE_RATIO = 0.97
# Looser snipe ratio for players about to be force-assigned to a random
# roster: claiming them deterministically beats the dump lottery.
_MEDIUM_DUMP_SNIPE_RATIO = 0.90
# Bid everything (minus slot reserve) on the highest-marginal remaining
# target: no better opportunity exists later in the queue.
_MEDIUM_ALL_IN_TOP = True


def _fast_team_scorer(missing_position_penalty: int):
    """Returns team_score(stat_tuples) mirroring Player.compute_score +
    best_lineup, but operating on plain tuples with a memoized position
    matching so it is fast enough for exhaustive combination search."""
    penalty = 1.0 / (2.0 ** (missing_position_penalty / 4.0))
    shortfall_memo: dict[tuple[int, ...], int] = {}
    caps = (
        _REQUIRED_POS_COUNTS['guard'],
        _REQUIRED_POS_COUNTS['forward'],
        _REQUIRED_POS_COUNTS['center'],
    )
    total_required = sum(caps)

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

    def team_score(stats: list[tuple]) -> float:
        pts = ast = reb = blk = stl = tov = tsm = tsa = 0.0
        masks = []
        for s in stats:
            pts += s[0]; ast += s[1]; reb += s[2]; blk += s[3]
            stl += s[4]; tov += s[5]
            tsm += s[6] * s[7]; tsa += s[7]
            masks.append(s[8])
        if tov <= 0:
            tov = 1e-9
        if tsa <= 0:
            tsa = 1e-9
        ts = tsm / tsa
        base = (pts * ast * reb * blk ** 0.2 * stl ** 0.2
                * (blk + stl) ** 0.4 * ts ** 1.5 / math.sqrt(tov))
        return base * penalty ** shortfall(tuple(masks))

    return team_score


def _stat_tuple(p: NBAPlayer) -> tuple:
    mask = (1 if p.guard else 0) | (2 if p.forward else 0) | (4 if p.center else 0)
    return (p.pts, p.ast, p.reb, p.blk, p.stl, p.tov, p.ts, p.tsa, mask)


def _rank_score(p: NBAPlayer) -> float:
    """Solo strength used only for ranking candidates. Smoothed so players
    with a zero stat (e.g. 0.0 blocks) don't all collapse to rank 0."""
    return (max(p.pts, 0.1) * max(p.ast, 0.1) * max(p.reb, 0.1)
            * max(p.blk, 0.05) ** 0.2 * max(p.stl, 0.05) ** 0.2
            * (p.blk + p.stl + 0.05) ** 0.4
            * max(p.ts, 0.1) ** 1.5 / math.sqrt(max(p.tov, 0.25)))


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
    pool = ranked_future[:_MEDIUM_POOL_SIZE]
    pool_ids = {p.pid for p in pool}
    for pos, need in _REQUIRED_POS_COUNTS.items():
        have = (sum(1 for p in pool if getattr(p, pos))
                + sum(1 for p in current_team if getattr(p, pos)))
        for cand in ranked_future:
            if have >= need:
                break
            if cand.pid not in pool_ids and getattr(cand, pos):
                pool.append(cand)
                pool_ids.add(cand.pid)
                have += 1

    team_score = _fast_team_scorer(missing_position_penalty)
    base_stats = [_stat_tuple(p) for p in current_team]
    cand_stats = [_stat_tuple(current)] + [_stat_tuple(p) for p in pool]
    all_indices = tuple(range(len(cand_stats)))
    CURRENT = 0

    def eval_team(extra_indices) -> float:
        return team_score(base_stats + [cand_stats[i] for i in extra_indices])

    def best_team(indices, choose, forced=()) -> tuple[float, tuple[int, ...]]:
        if choose <= 0 or not indices:
            return eval_team(forced), ()
        if len(indices) <= choose:
            return eval_team(tuple(forced) + tuple(indices)), tuple(indices)
        best_score, best_combo = -1.0, ()
        forced = tuple(forced)
        for combo in itertools.combinations(indices, choose):
            s = eval_team(forced + combo)
            if s > best_score:
                best_score, best_combo = s, combo
        return best_score, best_combo

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

def compute_bid_hard(
    missing_position_penalty: int,
    additional_players: int,
    room_players: list[Player],
    player_queue: list[NBAPlayer],
    current_team: list[NBAPlayer],
    balance: int,
) -> int:
    return 1


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
