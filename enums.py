from enum import Enum

class Failure(Enum):
    ROOM_CODE_NOT_FOUND = 1
    REQUIRES_OWNER = 2
    PLAYER_ID_NOT_FOUND = 3
    EMPTY_PLAYER_NAME = 4
    GAME_IN_PROGRESS = 5