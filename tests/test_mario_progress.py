"""Small ROM-backed checks for the wrapper's progress direction."""

from pathlib import Path
import os

import pytest
import numpy as np

from mario_env import MarioDeluxeEnv


ROM_PATH = Path(os.environ.get("MARIO_ROM", "roms/Super Mario Bros. Deluxe.gbc"))


@pytest.mark.parametrize("player_state", [0x05, 0x0A])
def test_flagpole_and_castle_states_are_success() -> None:
    """The wrapper's two level-end player states must be terminal success."""

    environment = object.__new__(MarioDeluxeEnv)
    environment._previous_mode = MarioDeluxeEnv.LEVEL_MODE

    assert environment._is_level_complete(player_state, MarioDeluxeEnv.LEVEL_MODE)


def test_non_victory_player_state_is_not_success() -> None:
    """A normal or death state must not be mistaken for reaching the goal."""

    environment = object.__new__(MarioDeluxeEnv)
    environment._previous_mode = MarioDeluxeEnv.LEVEL_MODE

    assert not environment._is_level_complete(0x00, MarioDeluxeEnv.LEVEL_MODE)
    assert not environment._is_level_complete(0x03, MarioDeluxeEnv.LEVEL_MODE)


def test_return_to_world_map_is_success() -> None:
    """The post-victory mode transition remains a valid completion signal."""

    environment = object.__new__(MarioDeluxeEnv)
    environment._previous_mode = MarioDeluxeEnv.LEVEL_MODE

    assert environment._is_level_complete(0x00, MarioDeluxeEnv.WORLD_MAP_MODE)


def test_stagnation_limit_requires_meaningful_forward_progress() -> None:
    """Small oscillations must not keep an episode alive indefinitely."""

    environment = object.__new__(MarioDeluxeEnv)
    environment._stagnation_steps = 0

    for _ in range(3):
        environment._update_stagnation(1)
    assert environment._stagnation_steps == 3

    environment._update_stagnation(environment.MIN_PROGRESS_ADVANCE)
    assert environment._stagnation_steps == 0


@pytest.mark.skipif(not ROM_PATH.exists(), reason="Super Mario Bros. Deluxe ROM is not available")
def test_object_metadata_is_included_in_observation_features() -> None:
    """The observation must expose the fixed-size object metadata channel."""

    environment = MarioDeluxeEnv(ROM_PATH)
    try:
        observation, _ = environment.reset(seed=7)
        assert observation["features"].shape == (len(environment.FEATURE_NAMES),)
        assert np.all(observation["features"] >= -1.0)
        assert np.all(observation["features"] <= 1.0)
    finally:
        environment.close()


@pytest.mark.skipif(not ROM_PATH.exists(), reason="Super Mario Bros. Deluxe ROM is not available")
def test_timer_uses_normal_binary_game_value() -> None:
    """The game timer must start at 400, not the wrapper's BCD value of 1024."""

    environment = MarioDeluxeEnv(ROM_PATH, max_episode_steps=1)
    try:
        environment.reset(seed=7)
        assert environment.mario.time_left == 400
    finally:
        environment.close()


@pytest.mark.skipif(not ROM_PATH.exists(), reason="Super Mario Bros. Deluxe ROM is not available")
def test_left_only_progress_reaches_the_left_wall(capsys: pytest.CaptureFixture[str]) -> None:
    """Holding left must move toward lower progress, not the level goal."""

    environment = MarioDeluxeEnv(ROM_PATH, frame_skip=2, frame_stack=12)
    try:
        environment.reset(seed=7)
        left_action = environment.ACTIONS.index(("left",))
        progress = []
        for _ in range(100):
            _, _, terminated, _, info = environment.step(left_action)
            progress.append(int(info["progress"]))
            if terminated:
                break
    finally:
        environment.close()

    print({"action": "left", "progress": progress})
    assert progress
    assert min(progress) < progress[0]
    assert max(progress) <= progress[0]


@pytest.mark.skipif(not ROM_PATH.exists(), reason="Super Mario Bros. Deluxe ROM is not available")
def test_right_only_progress_reaches_first_enemy() -> None:
    """Holding right must advance toward the first obstacle."""

    environment = MarioDeluxeEnv(ROM_PATH, frame_skip=2, frame_stack=12)
    try:
        environment.reset(seed=7)
        right_action = environment.ACTIONS.index(("right",))
        progress = []
        for _ in range(150):
            _, _, terminated, _, info = environment.step(right_action)
            progress.append(int(info["progress"]))
            if terminated:
                break
    finally:
        environment.close()

    print({"action": "right", "progress": progress})
    assert progress
    assert max(progress) > progress[0] + 100
    assert max(progress) >= 500
