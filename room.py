from __future__ import annotations

from game import Auction, NBAPlayer
from pydantic import BaseModel, Field
import random
import time
import os
import copy
import math
import pandas as pd
import bot_strategy

_TOP_PLAYERS_DF: pd.DataFrame = None
_REQUIRED_POS_COUNTS = {'guard': 2, 'forward': 2, 'center': 1}

def _load_players_pool():
    global _TOP_PLAYERS_DF
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'player_averages_2025_26.csv')
    
    df = pd.read_csv(csv_path)
    _TOP_PLAYERS_DF = df.head(150)

def get_sampled_players(num_players_needed: int) -> list[NBAPlayer]:
    global _TOP_PLAYERS_DF
    if _TOP_PLAYERS_DF is None:
        _load_players_pool()
    
    sample_size = min(num_players_needed, len(_TOP_PLAYERS_DF))
    sampled_df = _TOP_PLAYERS_DF.sample(n=sample_size)
    
    players = [
        NBAPlayer(
            name=f"{row.firstName} {row.lastName}",
            pid=int(row.personId),
            guard=bool(row.guard),
            forward=bool(row.forward),
            center=bool(row.center),
            pts=float(row.points),
            ast=float(row.assists),
            reb=float(row.reboundsTotal),
            blk=float(row.blocks),
            stl=float(row.steals),
            tov=float(row.turnovers),
            ts=float(row.ts)
        )
        for row in sampled_df.itertuples(index=False)
    ]
    return players

class Room(BaseModel):
    owner_id: str
    join_code: str
    bid_timer: int
    missing_position_penalty: int
    additional_players_queued: int

    members: dict[str, Player] = Field(default_factory=dict)
    player_queue: list[NBAPlayer] = Field(default_factory=list)
    round_num: int = 0
    current_auction: Auction | None = None
    prev_auction_result: PrevAuctionResult | None = None
    prev_game_final: list[Player] = Field(default_factory=list)

    @classmethod
    def create(
        cls, owner_id, owner_name, join_code, bid_timer,
        missing_position_penalty, additional_players_queued
    ) -> Room:
        room = cls(
            owner_id=owner_id,
            join_code=join_code,
            bid_timer=bid_timer,
            missing_position_penalty=missing_position_penalty,
            additional_players_queued=additional_players_queued)
        room.members[owner_id] = Player(name=owner_name)
        return room

    def add_member(self, player_id: str, player_name: str):
        self.members[player_id] = Player(name=player_name)

    def add_bot(self, player_id: str, player_name: str, difficulty: str):
        self.members[player_id] = Player(name=player_name, is_bot=True, bot_difficulty=difficulty)

    def has_bot_difficulty(self, difficulty: str) -> bool:
        return any(m.is_bot and m.bot_difficulty == difficulty for m in self.members.values())

    def default_bot_name(self, difficulty: str) -> str:
        """Default display name for a bot of the given difficulty."""
        return f"{difficulty.capitalize()} Bot"

    def next_round(self):
        self.round_num += 1
        self.current_auction = Auction(
            round_num=self.round_num,
            expected_player_ids=self._expected_bidders(),
            end_ts=time.time() + self.bid_timer
        )
        self._submit_bot_bids()

    def _submit_bot_bids(self):
        """Compute and record bids for every bot expected to bid this round.

        Each bot bids using the strategy for its difficulty. A room holds at
        most one bot per difficulty (three total), and only the ``hard``
        strategy is combinatorially expensive, so at most one costly valuation
        runs per round.
        """
        if self.current_auction is None:
            return
        additional_players = self.additional_players_queued * len(self.members)
        for player_id in self.current_auction.expected_player_ids:
            member = self.members.get(player_id)
            if member is None or not member.is_bot:
                continue
            bid = bot_strategy.compute_bid(
                member.bot_difficulty,
                player_queue=self.player_queue,
                missing_position_penalty=self.missing_position_penalty,
                additional_players=additional_players,
                balance=member.balance,
                current_team=member.nba_team,
            )
            self.current_auction.bids[player_id] = max(0, min(bid, member.balance))

    def start_game(self):
        num_players_needed = (5 + self.additional_players_queued) * len(self.members)
        self.player_queue = get_sampled_players(num_players_needed)
        # This reset step isn't done on game completion in order to preserve history
        self.prev_auction_result = None
        self.next_round()

    def handle_auction_end(self, winner_id: str, price_paid: int):
        nba_player = self.player_queue[0]
        if not winner_id:
            skip_threshold = self.additional_players_queued * len(self.members)
            if nba_player.skipped > skip_threshold:
                self.handle_auction_end(
                    random.choice(self._incomplete_roster_members()),
                    price_paid # Should be 0 in this case
                )
                return
            nba_player.skipped += 1
            self.player_queue.append(self.player_queue.pop(0))
            winner_name = ""
        elif winner_id in self.members:
            winner = self.members[winner_id]
            winner.nba_team.append(self.player_queue.pop(0))
            winner.balance -= price_paid
            unfilled_positions = winner.best_lineup() # Call best_lineup() after appending the new player to nba_team
            winner.compute_score(unfilled_positions, 1 / 2 ** (self.missing_position_penalty / 4))
            winner_name = winner.name

        self.prev_auction_result = PrevAuctionResult(
            winner=winner_name,
            nba_player=nba_player,
            price_paid=price_paid
        )
        if not self._incomplete_roster_members():
            self.prev_game_final = copy.deepcopy(list(self.members.values()))
            self._game_finished_reset()
        else:
            self.next_round()

    def _incomplete_roster_members(self) -> tuple[str]:
        return tuple(k for k in self.members if len(self.members[k].nba_team) < 5)

    def _expected_bidders(self) -> set[str]:
        return set(k for k in self.members if len(self.members[k].nba_team) < 5 and self.members[k].balance > 0)

    def _game_finished_reset(self):
        self.round_num = 0
        self.current_auction = None
        for player in self.members.values():
            player.nba_team = []
            player.lineup = dict()
            player.avg_stats = dict()
            player.balance = 100
            player.score = 0.0

class Player(BaseModel):
    name: str
    is_bot: bool = False
    bot_difficulty: str | None = None
    nba_team: list[NBAPlayer] = Field(default_factory=list)
    lineup: dict[str, list[int]] = Field(default_factory=dict)
    avg_stats: dict[str, float] = Field(default_factory=dict)
    balance: int = 100
    score: float = 0.0

    def compute_score(self, unfilled_positions: int, penalty: float) -> float:
        pts, ast, reb, blk, stl, tov, ts = [0] * 7
        for player in self.nba_team:
            pts += player.pts
            ast += player.ast
            reb += player.reb
            blk += player.blk
            stl += player.stl
            tov += player.tov
            ts += player.ts
        score = pts * ast * reb * blk ** 0.2 * stl ** 0.2 * math.sqrt(blk + stl) * ts ** 1.5 / math.sqrt(tov) * penalty ** unfilled_positions
        self.score = score
        c = len(self.nba_team)
        self.avg_stats = {
            'pts': pts/c, 'ast': ast/c, 'reb': reb/c, 'blk': blk/c,
            'stl': stl/c, 'tov': tov/c, 'ts': ts/c
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

class PrevAuctionResult(BaseModel):
    winner: str
    nba_player: NBAPlayer
    price_paid: int