from game import Auction, NBAPlayer
from player import Player
from pydantic import BaseModel, Field
import random
import time
import os
import copy
import pandas as pd
import bot_strategy

SUPPORTED_ERAS = {
    'averages_1990_00', 'averages_2000_10', 'averages_2010_20', 'averages_2020_26', 'averages_2025_26',
    'peaks_1990_00', 'peaks_2000_10'
}
_TOP_PLAYERS_BY_ERA: dict[str, pd.DataFrame] = dict()

def _load_players_pool(nba_era: str):
    global _TOP_PLAYERS_BY_ERA

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, f"player_{nba_era}.csv")
    
    df = pd.read_csv(csv_path).head(150)
    df['br_id'] = df['br_id'].fillna('')
    _TOP_PLAYERS_BY_ERA[nba_era] = df
    return df

def get_sampled_players(nba_era: str, num_players_needed: int) -> list[NBAPlayer]:
    global _TOP_PLAYERS_BY_ERA
    if nba_era in _TOP_PLAYERS_BY_ERA:
        era_df = _TOP_PLAYERS_BY_ERA[nba_era]
    else:
        era_df = _load_players_pool(nba_era)
    sample_size = min(num_players_needed, len(era_df))
    sampled_df = era_df.sample(n=sample_size)
    
    peak = 'peak' in nba_era
    players = []
    for row in sampled_df.itertuples(index=False):
        data = {
            'name': f"{row.firstName} {row.lastName}",
            'pid': int(row.personId),
            'br_id': str(row.br_id),
            'guard': bool(row.guard),
            'forward': bool(row.forward),
            'center': bool(row.center),
            'pts': float(row.points),
            'ast': float(row.assists),
            'reb': float(row.reboundsTotal),
            'blk': float(row.blocks),
            'stl': float(row.steals),
            'tov': float(row.turnovers),
            'ts': float(row.ts),
            'tsa': float(row.trueShootingAttempts)
        }
        if peak:
            data['peak'] = int(row.season)
        players.append(NBAPlayer(**data))
    return players

class Room(BaseModel):
    owner_id: str
    join_code: str
    bid_timer: int
    missing_position_penalty: int
    additional_players_queued: int
    nba_era: str

    members: dict[str, Player] = Field(default_factory=dict)
    player_queue: list[NBAPlayer] = Field(default_factory=list)
    round_num: int = 0
    current_auction: Auction | None = None
    prev_auction_result: PrevAuctionResult | None = None
    prev_game_final: list[Player] = Field(default_factory=list)

    @classmethod
    def create(
        cls, owner_id, owner_name, join_code, bid_timer,
        missing_position_penalty, additional_players_queued, nba_era
    ) -> Room:
        room = cls(
            owner_id=owner_id,
            join_code=join_code,
            bid_timer=bid_timer,
            missing_position_penalty=missing_position_penalty,
            additional_players_queued=additional_players_queued,
            nba_era=nba_era)
        room.members[owner_id] = Player(name=owner_name)
        return room

    def add_member(self, player_id: str, player_name: str):
        self.members[player_id] = Player(name=player_name)

    def add_bot(self, player_id: str, player_name: str, difficulty: str):
        self.members[player_id] = Player(name=player_name, bot_difficulty=difficulty)

    def has_bot_difficulty(self, difficulty: str) -> bool:
        return any(m.bot_difficulty == difficulty for m in self.members.values())

    def default_bot_name(self, difficulty: str) -> str:
        """Default display name for a bot of the given difficulty."""
        return f"{difficulty.capitalize()}Bot"

    def next_round(self):
        self.round_num += 1
        self.current_auction = Auction(
            round_num=self.round_num,
            expected_player_ids=self._expected_bidders(),
            end_ts=time.time() + self.bid_timer
        )
        self._submit_bot_bids()

    def _submit_bot_bids(self):
        """Compute and record bids for every bot expected to bid this round."""
        additional_players = self.additional_players_queued * len(self.members)
        for player_id in self.current_auction.expected_player_ids:
            member = self.members.get(player_id)
            if member is None or member.bot_difficulty is None:
                continue
            bot_inputs = bot_strategy.BotInputs(
                missing_position_penalty=self.missing_position_penalty,
                additional_players=additional_players,
                room_players=list(self.members.values()),
                player_queue=self.player_queue,
                bot_name=member.name,
                current_team=member.nba_team,
                balance=member.balance,
            )
            bid = bot_strategy.compute_bid(member.bot_difficulty, bot_inputs)
            self.current_auction.bids[player_id] = max(0, min(bid, member.balance))

    def start_game(self):
        num_players_needed = (5 + self.additional_players_queued) * len(self.members)
        self.player_queue = get_sampled_players(self.nba_era, num_players_needed)
        # This reset step isn't done on game completion in order to preserve history
        self.prev_auction_result = None
        self.next_round()

    def handle_auction_end(self, winner_id: str, price_paid: int):
        nba_player = self.player_queue[0]
        if not winner_id:
            incompletes = self._incomplete_roster_members()
            skip_threshold = self.additional_players_queued * len(self.members)
            if nba_player.skipped > skip_threshold or (incompletes and not self._expected_bidders()):
                self.handle_auction_end(
                    random.choice(incompletes),
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
            # Call best_lineup() after appending the new player to nba_team
            unfilled_positions = winner.best_lineup() if self.missing_position_penalty else 0
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

class PrevAuctionResult(BaseModel):
    winner: str
    nba_player: NBAPlayer
    price_paid: int