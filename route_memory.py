"""Episode-local route memory for the Deluxe environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RouteMemory:
    """Remember only the part of the route discovered during one episode."""

    player_left_margin: int = 16
    blocked_left_threshold: int = 8
    tile_map: dict[tuple[int, int], int] = field(default_factory=dict)
    min_progress: int = 0
    max_progress: int = 0
    max_camera_x: int = 0
    current_camera_x: int = 0
    retreat_boundary: int = 0
    blocked_left_steps: int = 0

    def reset(self, progress: int, camera_x: int) -> None:
        self.tile_map.clear()
        self.min_progress = progress
        self.max_progress = progress
        self.max_camera_x = camera_x
        self.current_camera_x = camera_x
        self.retreat_boundary = camera_x + self.player_left_margin
        self.blocked_left_steps = 0

    def observe(
        self,
        background: np.ndarray,
        *,
        camera_x: int,
        section_x: int,
        section_y: int,
        progress: int,
        sprite_offset: int,
    ) -> None:
        """Add the currently visible static tiles and update the forward frontier."""

        self.min_progress = min(self.min_progress, progress)
        self.max_progress = max(self.max_progress, progress)
        self.max_camera_x = max(self.max_camera_x, camera_x)
        self.current_camera_x = camera_x
        self.retreat_boundary = max(self.retreat_boundary, self.max_camera_x + self.player_left_margin)
        camera_tile_x = camera_x // 8
        for row, column in np.ndindex(background.shape):
            tile = int(background[row, column])
            if 0 < tile < sprite_offset:
                self.tile_map[(camera_tile_x + section_x + column, section_y + row)] = tile

    def record_motion(self, progress_delta: int, *, moving_left: bool, in_level: bool) -> None:
        if not in_level or not moving_left:
            self.blocked_left_steps = 0
            return
        if progress_delta <= -1:
            self.blocked_left_steps = 0
        else:
            self.blocked_left_steps += 1

    def features(self, progress: int) -> tuple[float, float, float, float]:
        """Return route context normalized for the PPO feature encoder."""

        return (
            np.clip((self.max_progress - progress) / 512.0, 0.0, 1.0),
            np.clip((progress - self.retreat_boundary) / 256.0, 0.0, 1.0),
            float(self.blocked_left_steps >= self.blocked_left_threshold),
            np.clip((self.max_camera_x - self.current_camera_x) / 512.0, 0.0, 1.0),
        )
