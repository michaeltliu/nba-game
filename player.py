from game import NBAPlayer
from pydantic import BaseModel, Field
import math

_REQUIRED_POS_COUNTS = {'guard': 2, 'forward': 2, 'center': 1}

class Player(BaseModel):
    name: str
    bot_difficulty: str | None = None
    nba_team: list[NBAPlayer] = Field(default_factory=list)
    lineup: dict[str, list[int]] = Field(default_factory=dict)
    avg_stats: dict[str, float] = Field(default_factory=dict)
    balance: int = 100
    score: float = 0.0

    def compute_score(self, unfilled_positions: int, penalty: float) -> float:
        pts, ast, reb, blk, stl, tov, tsm, tsa = [0] * 8
        for player in self.nba_team:
            pts += player.pts
            ast += player.ast
            reb += player.reb
            blk += player.blk
            stl += player.stl
            tov += player.tov
            tsm += player.ts * player.tsa
            tsa += player.tsa
        ts = tsm/tsa
        score = pts * ast * reb * blk ** 0.2 * stl ** 0.2 * (blk + stl) ** 0.4 * ts ** 1.5 / math.sqrt(tov) * penalty ** unfilled_positions
        self.score = score
        c = len(self.nba_team)
        self.avg_stats = {
            'pts': pts/c, 'ast': ast/c, 'reb': reb/c, 'blk': blk/c,
            'stl': stl/c, 'tov': tov/c, 'ts': ts
        }
        return score

    def best_lineup(self) -> int:
        """
        Assigns each player to at most one position slot via maximum bipartite
        matching, minimizing total shortfall = sum over categories of
        max(0, required - filled). The shortfall drives the position-bonus
        penalty multiplier in compute_score.

        Populates self.lineup: dict mapping 'guard'/'forward'/'center' -> list
        of indices into self.nba_team. Every player appears under exactly one
        position -- the one they're contributing the bonus toward, or their
        first eligible position if the requirement was already filled by
        someone else.

        Returns the total shortfall (0 = fully satisfied).
        """
        occupants = {pos: [] for pos in _REQUIRED_POS_COUNTS}

        def eligible_positions(idx):
            player = self.nba_team[idx]
            return [pos for pos in ('guard', 'forward', 'center') if getattr(player, pos)]

        def try_assign(idx, visited):
            for pos in eligible_positions(idx):
                if pos in visited:
                    continue
                visited.add(pos)
                if len(occupants[pos]) < _REQUIRED_POS_COUNTS[pos]:
                    occupants[pos].append(idx)
                    return True
                for occupant_idx in occupants[pos]:
                    if try_assign(occupant_idx, visited):
                        occupants[pos].remove(occupant_idx)
                        occupants[pos].append(idx)
                        return True
            return False

        for idx in range(len(self.nba_team)):
            try_assign(idx, set())

        counted = set(i for v in occupants.values() for i in v)
        shortfall = sum(_REQUIRED_POS_COUNTS.values()) - len(counted)

        lineup = {pos: list(occupants[pos]) for pos in _REQUIRED_POS_COUNTS}
        for idx, player in enumerate(self.nba_team):
            if idx in counted:
                continue
            for pos in ('guard', 'forward', 'center'):
                if getattr(player, pos):
                    lineup[pos].append(idx)
                    break

        self.lineup = lineup
        return shortfall