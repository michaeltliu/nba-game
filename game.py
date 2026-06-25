from dataclasses import dataclass
import time

@dataclass
class AuctionResult:
    auction_num: int
    resolved: bool
    winner_id: str = None
    price_paid: int = None

class Auction:
    def __init__(self, round_num: int, expected_player_ids: set[str], end_ts: int):
        self.round_num = round_num
        self.expected_player_ids = expected_player_ids
        self.end_ts = end_ts
        
        self.bids: dict[str, int] = dict()

    def maybe_resolve(self) -> AuctionResult:
        expired = time.time() > self.end_ts
        all_submitted = self.expected_player_ids <= self.bids.keys()
        if expired or all_submitted:
            return self._resolve()
        return AuctionResult(self.round_num, False)

    def _resolve(self) -> AuctionResult:
        first_price = 0
        second_price = 0
        winner_id = ""
        for player_id, bid in self.bids.items():
            if bid > first_price:
                second_price = first_price
                first_price = bid
                winner_id = player_id
            elif bid > second_price:
                second_price = bid
        return AuctionResult(self.round_num, True, winner_id, second_price)

@dataclass
class NBAPlayer:
    name: str
    pid: int
    pts: float
    ast: float
    reb: float
    blk: float
    stl: float
    tov: float
    ts: float
    skipped: bool = False