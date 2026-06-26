from fastapi import FastAPI, Header
from pydantic import BaseModel
from enums import Failure
from room import Room
import redis_store
import uuid


app = FastAPI()

class CreateRoomRequest(BaseModel):
    player_name: str
    bid_submission_timer: int
    missing_position_penalty: int

@app.post("/create-room")
async def create_room(request: CreateRoomRequest):
    if not request.player_name:
        return {'success': False, 'failure_msg': Failure.EMPTY_PLAYER_NAME.name}
    owner_id = str(uuid.uuid4())
    penalty = request.missing_position_penalty if request.missing_position_penalty in (0, 1, 2) else 1
    bid_timer = max(request.bid_submission_timer, 10)
    room = await redis_store.create_room_with_unique_code(
        lambda code: Room.create(owner_id, request.player_name, code, bid_timer, penalty)
    )
    return {'success': True, 'room_code': room.join_code, 'player_id': owner_id}


class JoinRoomRequest(BaseModel):
    player_name: str

@app.post("/join-room/{room_code}")
async def join_room(room_code: str, request: JoinRoomRequest):
    if not request.player_name:
        return {'success': False, 'failure_msg': Failure.EMPTY_PLAYER_NAME.name}
    async with redis_store.room_lock(room_code):
        room = await redis_store.load_room(room_code)
        if room is None:
            return {'success': False, 'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name}
        if room.round_num > 0:
            return {'success': False, 'failure_msg': 'GAME_IN_PROGRESS'}
        player_id = str(uuid.uuid4())
        room.add_member(player_id, request.player_name)
        await redis_store.save_room(room)
    return {'success': True, 'player_id': player_id}


@app.get("/rooms/{room_code}/status")
async def room_status(room_code: str):
    async with redis_store.room_lock(room_code):
        room = await redis_store.load_room(room_code)
        if room is None:
            return {'success': False, 'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name}
        if room.round_num > 0:
            auction_result = room.current_auction.maybe_resolve()
            if auction_result.resolved:
                room.handle_auction_end(auction_result.winner_id, auction_result.price_paid)
                await redis_store.save_room(room)
        status = {
            'success': True,
            'members': list(room.members.values()),
            'player_queue': room.player_queue,
            'round_num': room.round_num,
            'round_ends_at': room.current_auction.end_ts if room.current_auction else 0,
            'bids_received': len(room.current_auction.bids) if room.current_auction else 0,
            'prev_auction_result': room.prev_auction_result,
            'prev_game_final': room.prev_game_final,
        }
    return status


@app.post("/rooms/{room_code}/start-game")
async def start_game(room_code: str, x_player_id: str = Header(...)):
    async with redis_store.room_lock(room_code):
        room = await redis_store.load_room(room_code)
        if room is None:
            return {'success': False, 'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name}
        if room.owner_id != x_player_id:
            return {'success': False, 'failure_msg': Failure.REQUIRES_OWNER.name}
        if room.round_num > 0:
            return {'success': False, 'failure_msg': 'ALREADY_STARTED'}
        room.start_game()
        await redis_store.save_room(room)
    return {'success': True}


class SubmitBidRequest(BaseModel):
    bid_amount: int
    round_num: int

@app.post("/rooms/{room_code}/bid")
async def submit_bid(
    room_code: str,
    request: SubmitBidRequest,
    x_player_id: str = Header(...)
):
    async with redis_store.room_lock(room_code):
        room = await redis_store.load_room(room_code)
        if room is None:
            return {'success': False, 'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name}
        if x_player_id not in room.members:
            return {'success': False, 'failure_msg': Failure.PLAYER_ID_NOT_FOUND.name}
        if request.round_num <= 0 or request.round_num != room.round_num:
            return {'success': False, 'failure_msg': 'BAD_ROUND_NUMBER'}
        bid = request.bid_amount
        if bid < 0 or bid > room.members[x_player_id].balance:
            return {'success': False, 'failure_msg': 'INVALID_BID_AMOUNT'}
        if len(room.members[x_player_id].nba_team) >= 5:
            return {'success': False, 'failure_msg': 'ROSTER_FULL'}
        room.current_auction.bids[x_player_id] = bid
        auction_result = room.current_auction.maybe_resolve()
        if auction_result.resolved:
            room.handle_auction_end(auction_result.winner_id, auction_result.price_paid)
        await redis_store.save_room(room)
    return {'success': True}
