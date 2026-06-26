from game import Auction, AuctionResult, NBAPlayer
from pydantic import BaseModel, Field
import random
import time
import os
import copy
import pandas as pd

_TOP_200_PLAYERS_DF: pd.DataFrame = None

def _load_players_pool():
    global _TOP_200_PLAYERS_DF
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'player_averages_2025_26.csv')
    
    df = pd.read_csv(csv_path)
    _TOP_200_PLAYERS_DF = df.head(200)

def get_sampled_players(num_players_needed: int) -> list[NBAPlayer]:
    global _TOP_200_PLAYERS_DF
    if _TOP_200_PLAYERS_DF is None:
        _load_players_pool()
                
    sample_size = min(num_players_needed, len(_TOP_200_PLAYERS_DF))
    sampled_df = _TOP_200_PLAYERS_DF.sample(n=sample_size)
    
    players = [
        NBAPlayer(
            name=f"{row.firstName} {row.lastName}",
            pid=int(row.personId),
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

    members: dict[str, Player] = Field(default_factory=dict)
    player_queue: list[NBAPlayer] = Field(default_factory=list)
    round_num: int = 0
    current_auction: Auction | None = None
    prev_auction_result: PrevAuctionResult | None = None
    prev_game_final: list[Player] = Field(default_factory=list)

    @classmethod
    def create(cls, owner_id, owner_name, join_code, bid_timer, missing_position_penalty) -> Room:
        room = cls(
            owner_id=owner_id,
            join_code=join_code,
            bid_timer=bid_timer,
            missing_position_penalty=missing_position_penalty)
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
        num_players_needed = 5 * len(self.members)
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
            winner.compute_score()
            winner_name = winner.name

        self.prev_auction_result = PrevAuctionResult(
            winner=winner_name,
            nba_player=nba_player,
            price_paid=price_paid
        )
        if not self.player_queue:
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
            player.balance = 100
            player.score = 0.0

class Player(BaseModel):
    name: str
    nba_team: list[NBAPlayer] = Field(default_factory=list)
    balance: int = 100
    score: float = 0.0

    def compute_score(self) -> float:
        pts, ast, reb, blk, stl, tov, ts = [0] * 7
        for player in self.nba_team:
            pts += player.pts
            ast += player.ast
            reb += player.reb
            blk += player.blk
            stl += player.stl
            tov += player.tov
            ts += player.ts
        score = pts * ast * reb * blk * stl * ts / (tov ** 0.5)
        self.score = score
        return score

class PrevAuctionResult(BaseModel):
    winner: str
    nba_player: NBAPlayer
    price_paid: int