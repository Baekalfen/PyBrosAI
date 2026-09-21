"""Checks for episode-local route memory."""

import numpy as np

from route_memory import RouteMemory


def test_route_memory_clears_between_episodes() -> None:
    memory = RouteMemory()
    memory.reset(progress=100, camera_x=80)
    memory.observe(
        np.asarray([[1, 0], [0, 2]], dtype=np.uint32),
        camera_x=80,
        section_x=0,
        section_y=0,
        progress=100,
        sprite_offset=0x100,
    )
    assert memory.tile_map

    memory.reset(progress=0, camera_x=0)

    assert memory.tile_map == {}
    assert memory.max_progress == 0
    assert memory.retreat_boundary == 16


def test_camera_frontier_makes_backtracking_one_way() -> None:
    memory = RouteMemory()
    memory.reset(progress=0, camera_x=0)
    memory.observe(
        np.zeros((2, 2), dtype=np.uint32),
        camera_x=256,
        section_x=0,
        section_y=0,
        progress=320,
        sprite_offset=0x100,
    )

    known_ahead, retreat_margin, _, camera_frontier_distance = memory.features(320)

    assert known_ahead == 0.0
    assert retreat_margin > 0.0
    assert camera_frontier_distance == 0.0
    assert memory.retreat_boundary == 272
