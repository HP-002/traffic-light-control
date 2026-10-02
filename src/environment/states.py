from enum import Enum

class LightState(Enum):
    ALL_RED     = 0
    NS_GREEN    = 1
    NS_YELLOw   = 2
    EW_GREEN    = 3
    EW_YELLOW   = 4

class TrafficState(Enum):
    EMPTY               = 0
    LIGHT_TRAFFIC       = 1
    MODERATE_TRAFFIC    = 2
    HEAVY_TRAFFIC       = 3