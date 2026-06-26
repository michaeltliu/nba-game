import os
import random
import string
import redis.asyncio as redis
from collections.abc import Callable
from redis.asyncio.lock import Lock
from room import Room

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
ROOM_TTL_SECONDS = 60 * 60  # rooms auto-expire 1hr after last write

r = redis.from_url(REDIS_URL, decode_responses=True)


def _room_key(room_code: str) -> str:
    return f"room:{room_code}"

def _lock_key(room_code: str) -> str:
    return f"lock:room:{room_code}"

async def load_room(room_code: str) -> Room | None:
    raw = await r.get(_room_key(room_code))
    return Room.model_validate_json(raw) if raw else None

async def save_room(room: Room):
    await r.set(_room_key(room.join_code), room.model_dump_json(), ex=ROOM_TTL_SECONDS)

async def create_room_with_unique_code(room_factory: Callable[[str], Room]) -> Room:
    for _ in range(10):
        room_code = ''.join(random.choices(string.ascii_uppercase, k=5))
        room = room_factory(room_code)
        claimed = await r.set(_room_key(room_code), room.model_dump_json(),
                               nx=True, ex=ROOM_TTL_SECONDS)
        if claimed:
            return room
    raise RuntimeError("Could not claim a unique room code after 10 attempts")

def room_lock(room_code: str) -> Lock:
    """Use as: async with room_lock(room_code): ..."""
    return r.lock(_lock_key(room_code), timeout=5)