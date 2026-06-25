from fastapi import FastAPI, Header
from pydantic import BaseModel
from enums import Failure
from room import Room
import random
import string
import uuid

rooms: dict[str, Room] = dict()

app = FastAPI()

class CreateRoomRequest(BaseModel):
    player_name: str
    bid_submission_timer: int
    missing_position_penalty: int

@app.post("/create-room")
async def create_room(request: CreateRoomRequest):
    if not request.player_name:
        return {
            'success': False,
            'failure_msg': Failure.EMPTY_PLAYER_NAME.name
        }
    room_code = ''.join(random.choices(string.ascii_uppercase, k=5))
    owner_id = str(uuid.uuid4())
    rooms[room_code] = Room(
        owner_id,
        request.player_name,
        room_code,
        max(request.bid_submission_timer, 10),
        request.missing_position_penalty if request.missing_position_penalty in (0, 1, 2) else 1)
    return {'success': True, 'room_code': room_code, 'player_id': owner_id}


class JoinRoomRequest(BaseModel):
    player_name: str

@app.post("/join-room/{room_code}")
async def join_room(room_code: str, request: JoinRoomRequest):
    if not request.player_name:
        return {
            'success': False,
            'failure_msg': Failure.EMPTY_PLAYER_NAME.name
        }
    if room_code not in rooms:
        return {
            'success': False,
            'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name
        }
    room = rooms[room_code]
    player_id = str(uuid.uuid4())
    room.add_member(player_id, request.player_name)
    print(room)
    return {'success': True, 'player_id': player_id}


@app.get("/rooms/{room_code}/status")
async def room_status(room_code: str):
    if room_code not in rooms:
        return {
            'success': False,
            'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name
        }
    room = rooms[room_code]
    if room.round_num > 0:
        auction_status = room.current_auction.maybe_resolve()
    status = {
        'success': True,
        'members': list(room.members.values()),
        'player_queue': room.player_queue,
        'round_num': room.round_num,
        'round_ends_at': room.current_auction.end_ts,
        'bids_received': len(room.current_auction.bids),
        'prev_auction_result': room.prev_auction_result
    }
    return status


@app.post("/rooms/{room_code}/start-game")
async def start_game(room_code: str, x_player_id: str = Header(...)):
    if room_code not in rooms:
        return {
            'success': False,
            'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name
        }
    room = rooms[room_code]
    if room.owner_id != x_player_id:
        return {
            'success': False,
            'failure_msg': Failure.REQUIRES_OWNER.name
        }
    if room.round_num > 0:
        return {
            'success': False,
            'failure_msg': 'ALREADY_STARTED'
        }
    room.start_game()
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
    if room_code not in rooms:
        return {
            'success': False,
            'failure_msg': Failure.ROOM_CODE_NOT_FOUND.name
        }
    room = rooms[room_code]
    if x_player_id not in room.members:
        return {
            'success': False,
            'failure_msg': Failure.PLAYER_ID_NOT_FOUND.name
        }
    if request.round_num != room.round_num:
        return {
            'success': False,
            'failure_msg': 'ROUND_MISMATCH'
        }
    bid = request.bid_amount
    if bid < 0 or bid > room.members[x_player_id].balance:
        return {
            'success': False,
            'failure_msg': 'INVALID_BID_AMOUNT'
        }
    auction = room.current_auction
    auction.bids[x_player_id] = bid
    auction_status = auction.maybe_resolve()
    if auction_status.resolved:
        room.handle_auction_end(
            auction_status.winner_id,
            auction_status.price_paid
        )
    return {'success': True}

