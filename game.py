from dataclasses import dataclass
import time

class Auction:
    def __init__(self, expected_player_ids: set[str], end_ts: int):
        self.expected_player_ids = expected_player_ids
        self.end_ts = end_ts
        
        self.bids = dict[str, int] = dict()

    def maybe_resolve(self) -> dict:
        expired = time.time() > self.end_ts
        all_submitted = self.expected_player_ids <= self.bids.keys()
        if expired or all_submitted:
            return self._resolve()
        return {'resolved': False}

    def _resolve(self) -> dict:
        first_price = 0
        second_price = 0
        winner_id = ""
        for player_id, bid in self.bids:
            if bid > first_price:
                second_price = first_price
                first_price = bid
                winner_id = player_id
            elif bid > second_price:
                second_price = bid
        return {'resolved': True, 'winner_id': winner_id, "price_paid": second_price}

@dataclass
class NBAPlayer:
    name: str
    pts: float
    ast: float
    reb: float
    blk: float
    stl: float
    tov: float
    ts: float