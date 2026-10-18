from enum import Enum

class LightState(Enum):
    ALL_RED     = 0
    N_GREEN     = 1
    N_YELLOW    = 2
    S_GREEN     = 3
    S_YELLOW    = 4
    E_GREEN     = 5
    E_YELLOW    = 6
    W_GREEN     = 7
    W_YELLOW    = 8

class TrafficState(Enum):
    EMPTY               = 0
    LIGHT_TRAFFIC       = 1
    MODERATE_TRAFFIC    = 2
    HEAVY_TRAFFIC       = 3

class ExtraTimeState(Enum):
    BELOW = 0
    ABOVE = 1
