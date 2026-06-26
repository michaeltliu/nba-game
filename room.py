from game import Auction, NBAPlayer
from itertools import product
from pydantic import BaseModel, Field
import random
import time
import os
import copy
import pandas as pd

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

    def next_round(self):
        self.round_num += 1
        self.current_auction = Auction(
            round_num=self.round_num,
            expected_player_ids=self._expected_bidders(),
            end_ts=time.time() + self.bid_timer
        )

    def start_game(self):
        num_players_needed = (5 + self.additional_players_queued) * len(self.members)
        self.player_queue = get_sampled_players(num_players_needed)
        # This reset step isn't done on game completion in order to preserve history
        self.prev_auction_result = None
        self.next_round()

    def handle_auction_end(self, winner_id: str, price_paid: int):
        nba_player = self.player_queue[0]
        if not winner_id:
            if nba_player.skipped:
                self.handle_auction_end(
                    random.choice(self._incomplete_roster_members()),
                    price_paid # Should be 0 in this case
                )
                return
            nba_player.skipped = True
            self.player_queue.append(self.player_queue.pop(0))
            winner_name = ""
        elif winner_id in self.members:
            winner = self.members[winner_id]
            winner.nba_team.append(self.player_queue.pop(0))
            winner.balance -= price_paid
            unfilled_positions = winner.best_lineup() # Call best_lineup() after appending the new player to nba_team
            winner.compute_score(unfilled_positions, 1 / 2 ** (self.missing_position_penalty / 2))
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
            player.balance = 100
            player.score = 0.0

class Player(BaseModel):
    name: str
    nba_team: list[NBAPlayer] = Field(default_factory=list)
    lineup: dict[str, list[int]] = Field(default_factory=dict)
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
        score = pts * ast * reb * blk ** 0.8 * stl ** 0.8 * ts ** 1.5 / (tov ** 0.5) * penalty ** unfilled_positions
        self.score = score
        return score

    def best_lineup(self) -> int:
        """
        Assigns each player to at most one position slot (guard/forward/center)
        to minimize the number of unmet requirements (>=2 guards, >=2 forwards,
        >=1 center). A player can only fill a slot they're eligible for, and
        each player fills at most one slot.

        Populates self.lineup dict mapping 'guard'/'forward'/'center' -> list of
        indices of players in self.nba_team placed there

        Returns
        - unfulfilled_count: how many of the 3 requirement categories are still unmet
        """
        options_per_player = []
        for player in self.nba_team:
            choices = []
            if player.guard:
                choices.append('guard')
            if player.forward:
                choices.append('forward')
            if player.center:
                choices.append('center')
            options_per_player.append(choices)

        best_assignment = None
        best_unfulfilled = 4

        for assignment in product(*options_per_player):
            counts = {'guard': 0, 'forward': 0, 'center': 0}
            for pos in assignment:
                counts[pos] += 1

            unfulfilled = len([pos for pos in counts if counts[pos] < _REQUIRED_POS_COUNTS[pos]])

            if unfulfilled < best_unfulfilled:
                best_unfulfilled = unfulfilled
                best_assignment = assignment

        lineup = {'guard': [], 'forward': [], 'center': []}
        for idx, pos in enumerate(best_assignment):
            lineup[pos].append(idx)
        self.lineup = lineup

        return best_unfulfilled


class PrevAuctionResult(BaseModel):
    winner: str
    nba_player: NBAPlayer
    price_paid: int