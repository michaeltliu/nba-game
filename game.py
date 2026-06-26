from pydantic import BaseModel, Field
import time

class AuctionResult(BaseModel):
    auction_num: int
    resolved: bool
    winner_id: str = None
    price_paid: int = None

class Auction(BaseModel):
    round_num: int
    expected_player_ids: set[str]
    end_ts: float
    bids: dict[str, int] = Field(default_factory=dict)

    def maybe_resolve(self) -> AuctionResult:
        expired = time.time() > self.end_ts
        all_submitted = self.expected_player_ids <= self.bids.keys()
        if expired or all_submitted:
            return self._resolve()
        return AuctionResult(auction_num=self.round_num, resolved=False)

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
        return AuctionResult(
            auction_num=self.round_num,
            resolved=True,
            winner_id=winner_id,
            price_paid=second_price
        )

class NBAPlayer(BaseModel):
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