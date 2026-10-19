from dataclasses import dataclass

@dataclass
class Car:
    """A car queued at the intersection.

    Attributes:
        waiting_time: steps spent queued while its side was red
    """
    waiting_time: int = 0
