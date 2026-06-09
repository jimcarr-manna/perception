"""Shared datatypes for 2D feet, BEV feet, and rigid transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetFoot2D:
    u: float
    v: float
    class_id: int
    score: float


@dataclass(frozen=True)
class DetBox2D:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float

    @property
    def bottom_left(self) -> tuple[float, float]:
        return self.x1, self.y2

    @property
    def bottom_centre(self) -> tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), self.y2

    @property
    def bottom_right(self) -> tuple[float, float]:
        return self.x2, self.y2


@dataclass(frozen=True)
class FootBEV:
    x: float
    y: float
    z: float
    class_id: int
    score: float
    lidar_refined: bool = False
    # Camera-only ground-plane position before any LiDAR range correction.
    # None when no LiDAR data was available (values equal x/y in that case).
    x_init: float | None = None
    y_init: float | None = None


@dataclass(frozen=True)
class SE3:
    """Rigid transform: point_in_base = R @ point_in_child + t."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> SE3:
        return SE3(R=R, t=t)
