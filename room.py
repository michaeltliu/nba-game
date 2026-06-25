from dataclasses import dataclass, field
from game import Auction, AuctionResult, NBAPlayer
import json
import random
import time

class Room:
    def __init__(
        self,
        owner_id: str,
        owner_name: str,
        join_code: str,
        bid_timer: int,
        missing_position_penalty: int
    ):
        self.owner_id = owner_id
        self.join_code = join_code
        self.bid_timer = bid_timer
        self.missing_position_penalty = missing_position_penalty

        self.members: dict[str, Player] = {owner_id: Player(owner_name)}
        self.player_queue: list[NBAPlayer] = []
        self.round_num = 0
        self.current_auction = Auction(0, set(), 0)
        self.prev_auction_result = dict()

    def __str__(self):
        return f"""
        Room(
            owner_id={self.owner_id},
            join_code={self.join_code},
            bid_timer={self.bid_timer},
            missing_position_penalty={self.missing_position_penalty},
            members={self.members}
            player_queue={self.player_queue}
            round_num={self.round_num}
        )"""

    def __repr__(self):
        return self.__str__()

    def add_member(self, player_id: str, player_name: str):
        self.members[player_id] = Player(player_name)

    def next_round(self):
        self.round_num += 1
        self.current_auction = Auction(
            self.round_num,
            set(self.members),
            time.time() + self.bid_timer
        )

    def start_game(self):
        
        self.next_round()

    def handle_auction_end(self, winner_id: str, price_paid: int):
        nba_player = self.player_queue[0]
        if not winner_id:
            if nba_player.skipped:
                self.handle_auction_end(
                    random.choice(tuple(self.members)),
                    price_paid # Should be 0 in this case
                )
                return
            nba_player.skipped = True
            self.player_queue.append(self.player_queue.pop(0))
            self.prev_auction_result['winner'] = ""
        elif winner_id in self.members:
            winner = self.members[winner_id]
            winner.nba_team.append(self.player_queue.pop(0))
            winner.balance -= price_paid
            self.prev_auction_result['winner'] = winner.name

        self.prev_auction_result['nba_player'] = nba_player
        self.prev_auction_result['price_paid'] = price_paid
        self.next_round()

@dataclass
class Player:
    name: str
    nba_team: list[NBAPlayer] = field(default_factory=list)
    balance: int = 100

    def get_composite_score(self) -> float:
        pass