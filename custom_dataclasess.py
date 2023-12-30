from dataclasses import dataclass, field


@dataclass
class Vehicle:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    landmarks: list
    orientation: float

@dataclass
class Trajectory:
    positions: list = field(default_factory=list)

    def add_position(self, position):
        self.positions.append(position)