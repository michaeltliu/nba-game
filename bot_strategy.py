import itertools
from game import NBAPlayer
from player import Player, _REQUIRED_POS_COUNTS
from pydantic import BaseModel

DIFFICULTIES = ("easy", "medium", "hard")
SUBSTITUTION_THRESHOLD = 0.84
MAX_CANDIDATES_RATIO = 6

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
    print("current player:", current_player.pid, current_player.name)

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
    print("target team:", [p.name for p in target_team])
    marginals = {
        t.pid: u_best - best_team(pool_minus(pool, t), k)[0]
        for t in target_team
    }
    marginals_total = sum(marginals.values())
    print("marginals:", marginals)
    if current_player.pid in target_ids:
        if not marginals_total:
            share = 1 / k
        else:
            share = marginals[current_player.pid] / marginals_total
        print("bid:", max(1, int(balance * share)))
        return max(1, int(balance * share))

    non_target_utilities = {
        p.pid: best_team_including(pool, k, p) for p in pool if p.pid not in target_ids
    }
    print("non-target utilities:", non_target_utilities)
    u_current = non_target_utilities[current_player.pid]
    print("u_current:", u_current)
    end_index = -additional_players if additional_players else None
    replacement_utilities = sorted(non_target_utilities.values(), reverse=True)[:end_index]
    print("replacement utilities:", replacement_utilities)
    u_avg = sum(replacement_utilities) / len(replacement_utilities) if replacement_utilities else 0

    if u_current < u_avg or u_current < SUBSTITUTION_THRESHOLD * u_best:
        return 0

    def vorp(u: float, avg: float) -> float:
        return max(0, u - avg)
    
    vorp_share = vorp(u_current, u_avg) / sum(vorp(u, u_avg) for u in replacement_utilities)
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
    return 1

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
