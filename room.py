from game import Auction, NBAPlayer
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
        self.current_auction = None

    def __str__(self):
        return f"""
        Room(
            owner_id={self.owner_id},
            join_code={self.join_code},
            bid_timer={self.bid_timer},
            missing_position_penalty={self.missing_position_penalty},
            members={self.members}
        )"""

    def __repr__(self):
        return self.__str__()

    def add_member(self, player_id: str, player_name: str):
        self.members[player_id] = Player(player_name)

    def next_round(self):
        self.round_num += 1
        self.current_auction = Auction(set(self.members), time.time() + self.bid_timer)

class Player:
    def __init__(self, name: str):
        self.name = name
        self.nba_team = []
        self.balance = 100

    def __str__(self):
        return f"Player(name={self.name}, nba_team={self.nba_team}, balance={self.balance})"

    def __repr__(self):
        return self.__str__()

    def get_composite_score(self) -> float:
        pass