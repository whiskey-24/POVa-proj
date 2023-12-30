from dataclasses import dataclass


@dataclass
class Vehicle:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    landmarks: list
    orientation: float
