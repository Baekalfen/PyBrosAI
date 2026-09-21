"""Gymnasium environment for Super Mario Bros. Deluxe in PyBoy."""

from __future__ import annotations

from collections import deque
import importlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

from route_memory import RouteMemory


class MarioDeluxeEnv(gym.Env[dict[str, np.ndarray], int]):
    """A compact action-space environment around PyBoy's Super Mario Deluxe game wrapper.

    A matching Super Mario Bros. Deluxe ROM must be supplied by the caller.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    ACTIONS = (
        ("right",),
        ("right", "a"),
        ("right", "b"),
        ("right", "a", "b"),
        ("left",),
        ("left", "a"),
        ("left", "b"),
        ("left", "a", "b"),
        (),
    )
    FEATURE_NAMES = (
        "progress_velocity",
        "progress_position",
        "max_progress_position",
        "score_delta",
        "coin_delta",
        "time_delta",
        "player_y_velocity",
        "object_count",
        "hazard_count",
        "platform_count",
        "nearest_hazard_x",
        "nearest_hazard_y",
        "nearest_hazard_x_speed",
        "nearest_hazard_y_speed",
        "nearest_hazard_type",
        "nearest_platform_x",
        "nearest_platform_y",
        "nearest_platform_x_speed",
        "nearest_platform_y_speed",
        "nearest_platform_type",
        "player_y_position",
        "support_current",
        "support_ahead_2",
        "support_ahead_4",
        "support_ahead_6",
        "support_ahead_8",
        "solid_above_current",
        "solid_above_2",
        "solid_above_4",
        "solid_above_6",
        "solid_above_8",
        "route_known_ahead",
        "retreat_margin",
        "retreat_boundary_contact",
        "camera_frontier_distance",
    )
    LEVEL_MODE = 0x0B
    OVERWORLD_INIT_MODE = 0x04
    WORLD_MAP_MODE = 0x05
    LEVEL_TIME_LIMIT = 400
    # The disassembly names player state 05 FlagVictory_Main and state 0A
    # AxeVictory_Main. Both are entered only after reaching a level endpoint.
    LEVEL_END_STATES = frozenset((0x05, 0x0A))
    PLAYER_STATE_ADDRESS = 0xC1C1
    MODE_ADDRESS = 0xFFB5
    LEVEL_START_MAX_WAIT_FRAMES = 600
    SCORE_REWARD_SCALE = 0.02
    MAX_SCORE_EVENT_REWARD = 20.0
    COIN_REWARD_SCALE = 2.0
    BACKWARD_PROGRESS_PENALTY = 0.001
    GAP_APPROACH_PENALTY = 0.5
    GAP_JUMP_REWARD = 0.1
    MIN_PROGRESS_ADVANCE = 8
    DEFAULT_STAGNATION_LIMIT = 450
    HAZARD_MAPPED_IDS = frozenset((7, 10, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 30, 31))
    PLATFORM_MAPPED_IDS = frozenset((16, 21))
    PLAYER_GAME_AREA_X = 8.0
    PLAYER_GAME_AREA_Y = 14.0

    def __init__(
        self,
        rom_path: str | Path,
        *,
        frame_skip: int = 2,
        frame_stack: int = 12,
        max_episode_steps: int = 6_000,
        max_stagnation_steps: int = DEFAULT_STAGNATION_LIMIT,
        curriculum_target: int | None = None,
        render_mode: str | None = None,
        world_level: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        if frame_skip < 1 or frame_stack < 1 or max_episode_steps < 1 or max_stagnation_steps < 1:
            raise ValueError("frame_skip, frame_stack, max_episode_steps, and max_stagnation_steps must be positive")

        self.rom_path = str(rom_path)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.max_stagnation_steps = max_stagnation_steps
        self.curriculum_target = curriculum_target
        self.render_mode = render_mode
        self.world_level = world_level
        self.pyboy = PyBoy(
            self.rom_path,
            window="GLFW" if render_mode == "human" else "null",
            sound_emulated=False,
            debug=render_mode == "human",
        )
        # Menu navigation in start_game performs hundreds of emulator ticks.
        # Keep it uncapped and apply demo realtime pacing after the level loads.
        self.pyboy.set_emulation_speed(0)
        self.mario = self.pyboy.game_wrapper

        expected_title = "MARIO DELUXAHY"
        if self.pyboy.cartridge_title != expected_title:
            self.pyboy.stop(save=False)
            raise ValueError(
                f"ROM has title {self.pyboy.cartridge_title!r}; "
                f"expected the Super Mario Bros. Deluxe ROM ({expected_title!r})"
            )

        mapping = np.asarray(self.mario.mapping_minimal, dtype=np.uint32)
        wrapper_module = importlib.import_module(type(self.mario).__module__)
        if not callable(getattr(self.mario, "object_slots", None)):
            self.pyboy.stop(save=False)
            raise RuntimeError("The mario_deluxe wrapper must provide object_slots()")
        self.player_y_address = int(getattr(wrapper_module, "ADDR_PLAYER_Y", 0xFFA9))
        self.player_x_address = int(getattr(wrapper_module, "ADDR_PLAYER_X", 0xC1CA))
        sprite_offset = int(getattr(wrapper_module, "DEFAULT_SPRITE_OFFSET", 0x100))
        if sprite_offset == 0:
            sprite_offset = 0x100
        self.wrapper_name = type(self.mario).__name__
        self.sprite_offset = sprite_offset
        self.height, self.width = self.mario.shape[1], self.mario.shape[0]
        self.num_tiles = int(np.max(mapping)) + sprite_offset + 1
        self.observation_space = spaces.Box(
            low=0, high=self.num_tiles - 1, shape=(frame_stack, self.height, self.width), dtype=np.uint32
        )
        self.observation_space = spaces.Dict(
            {
                "tiles": self.observation_space,
                "features": spaces.Box(low=-1.0, high=1.0, shape=(len(self.FEATURE_NAMES),), dtype=np.float32),
            }
        )
        self.feature_count = len(self.FEATURE_NAMES)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._previous_progress = 0
        self._max_progress = 0
        self._stagnation_steps = 0
        self._previous_score = 0
        self._previous_coins = 0
        self._previous_mode = self.LEVEL_MODE
        self._previous_player_y = 0
        self._episode_steps = 0
        self._level_start_progress = 0
        self._started = False
        self.route_memory = RouteMemory()

    def _observation(self, features: np.ndarray) -> dict[str, np.ndarray]:
        frame = np.asarray(self.mario.game_area(), dtype=np.uint32)
        if frame.shape != (self.height, self.width):
            raise RuntimeError(f"Unexpected game-area shape: {frame.shape}")
        self._frames.append(frame.copy())
        while len(self._frames) < self.frame_stack:
            self._frames.appendleft(frame.copy())
        return {"tiles": np.stack(tuple(self._frames), axis=0), "features": features}

    def _set_training_state(self) -> None:
        # Extra lives keep the emulator usable after reset while game_over
        # still terminates the current learning episode.
        self.mario.set_lives_left(99)
        self.mario.set_time_left(self.LEVEL_TIME_LIMIT)

    def _observe_route(self, progress: int) -> None:
        section_x, section_y, _, _ = self.mario.game_area_section
        background = np.asarray(self.mario.game_area_background(), dtype=np.uint32)
        self.route_memory.observe(
            background,
            camera_x=int(self.mario._camera_x()),
            section_x=section_x,
            section_y=section_y,
            progress=progress,
            sprite_offset=self.sprite_offset,
        )

    def _wait_for_level_start(self) -> None:
        """Skip the scripted level intro and stop at Mario's initial fall."""

        stable_ground_frames = 0
        for _ in range(self.LEVEL_START_MAX_WAIT_FRAMES):
            mode = int(self.pyboy.memory[self.MODE_ADDRESS])
            player_state = int(self.pyboy.memory[self.PLAYER_STATE_ADDRESS])
            player_x = int.from_bytes(
                self.pyboy.memory[self.player_x_address : self.player_x_address + 2], "little"
            )
            player_y = int(self.pyboy.memory[self.player_y_address])
            if mode == self.LEVEL_MODE and player_state == 0 and player_x <= 64:
                if player_y < 208:
                    return
                if player_y == 208:
                    stable_ground_frames += 1
                    if stable_ground_frames >= 10:
                        return
                else:
                    stable_ground_frames = 0
            else:
                stable_ground_frames = 0
            assert self.pyboy.tick(1, self.render_mode == "human", False)
        raise RuntimeError("Timed out waiting for the scripted intro to reach the playable level start")

    @staticmethod
    def _signed_byte_delta(current: int, previous: int) -> int:
        return (current - previous + 128) % 256 - 128

    def _nearest_object_features(self, objects: list[dict[str, Any]], mapped_ids: frozenset[int]) -> list[float]:
        candidates = [object_ for object_ in objects if int(object_.get("mapped_id", 0)) in mapped_ids]
        if not candidates:
            return [0.0] * 5
        object_ = min(
            candidates,
            key=lambda value: (
                (float(value.get("game_area_x", self.PLAYER_GAME_AREA_X)) - self.PLAYER_GAME_AREA_X) ** 2
                + (float(value.get("game_area_y", self.PLAYER_GAME_AREA_Y)) - self.PLAYER_GAME_AREA_Y) ** 2
            ),
        )
        return [
            np.clip((float(object_["game_area_x"]) - self.PLAYER_GAME_AREA_X) / 10.0, -1.0, 1.0),
            np.clip((float(object_["game_area_y"]) - self.PLAYER_GAME_AREA_Y) / 10.0, -1.0, 1.0),
            np.clip(float(object_["x_speed"]) / 8.0, -1.0, 1.0),
            np.clip(float(object_["y_speed"]) / 8.0, -1.0, 1.0),
            np.clip(float(object_.get("mapped_id", 0)) / 31.0, 0.0, 1.0),
        ]

    def _support_features(self) -> tuple[float, list[float], list[float]]:
        """Describe solid tiles below and above Mario in visible forward columns."""

        area = np.asarray(self.mario.game_area(), dtype=np.uint32)
        player_x = int.from_bytes(self.pyboy.memory[self.player_x_address : self.player_x_address + 2], "little")
        player_y = int(self.pyboy.memory[self.player_y_address])
        camera_x = int(self.mario._camera_x())
        section_x, section_y, _, _ = self.mario.game_area_section
        game_area_x = ((player_x - camera_x + 4) // 8) - section_x
        game_area_y = ((player_y + 4) // 8) - section_y
        row_start = min(max(game_area_y + 2, 0), area.shape[0])
        row_end = min(max(game_area_y, 0), area.shape[0])
        support = []
        solid_above = []
        for offset in (0, 2, 4, 6, 8):
            column = game_area_x + offset
            if column < 0 or column >= area.shape[1] or row_start >= area.shape[0]:
                support.append(0.0)
                solid_above.append(0.0)
                continue
            column_tiles = area[row_start:, column]
            support.append(float(np.any((column_tiles > 0) & (column_tiles < self.sprite_offset))))
            above_tiles = area[:row_end, column]
            solid_above.append(float(np.any((above_tiles > 0) & (above_tiles < self.sprite_offset))))
        return np.clip(player_y / 255.0, 0.0, 1.0), support, solid_above

    def _state_features(
        self,
        *,
        progress: int,
        max_progress: int,
        progress_delta: int,
        score_delta: int,
        coin_delta: int,
        time_delta: int,
        player_y_delta: int,
    ) -> np.ndarray:
        objects = list(self.mario.object_slots())
        hazards = [object_ for object_ in objects if int(object_.get("mapped_id", 0)) in self.HAZARD_MAPPED_IDS]
        platforms = [object_ for object_ in objects if int(object_.get("mapped_id", 0)) in self.PLATFORM_MAPPED_IDS]
        player_y_position, support, solid_above = self._support_features()
        return np.asarray(
            [
                np.clip(progress_delta / 16.0, -1.0, 1.0),
                np.clip(progress / 3300.0, 0.0, 1.0),
                np.clip(max_progress / 3300.0, 0.0, 1.0),
                np.clip(score_delta / 100.0, -1.0, 1.0),
                np.clip(coin_delta, -1.0, 1.0),
                np.clip(time_delta / 4.0, -1.0, 1.0),
                np.clip(player_y_delta / 8.0, -1.0, 1.0),
                np.clip(len(objects) / 15.0, 0.0, 1.0),
                np.clip(len(hazards) / 15.0, 0.0, 1.0),
                np.clip(len(platforms) / 15.0, 0.0, 1.0),
                *self._nearest_object_features(objects, self.HAZARD_MAPPED_IDS),
                *self._nearest_object_features(objects, self.PLATFORM_MAPPED_IDS),
                player_y_position,
                *support,
                *solid_above,
                *self.route_memory.features(progress),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        timer_div = None if seed is None else seed & 0xFF
        if not self._started:
            self.mario.start_game(world_level=self.world_level, timer_div=timer_div)
            self._started = True
        else:
            self.mario.reset_game(timer_div=timer_div)
        self._set_training_state()
        if self.render_mode == "human":
            self.pyboy.set_emulation_speed(1)
        # Refresh the wrapper's cached timer and other RAM-backed fields after
        # changing the training state.
        assert self.pyboy.tick(1, self.render_mode == "human", False)
        self._wait_for_level_start()
        self._frames.clear()
        self._level_start_progress = int(self.mario.level_progress)
        self._previous_progress = 0
        self._max_progress = self._previous_progress
        self._stagnation_steps = 0
        self._previous_score = int(self.mario.score)
        self._previous_coins = int(self.mario.coins)
        self._previous_time = int(self.mario.time_left)
        self._previous_player_y = int(self.pyboy.memory[self.player_y_address])
        self._previous_mode = int(self.pyboy.memory[self.MODE_ADDRESS])
        self._episode_steps = 0
        self.route_memory.reset(self._previous_progress, int(self.mario._camera_x()))
        self._observe_route(self._previous_progress)
        return self._observation(
            self._state_features(
                progress=self._previous_progress,
                max_progress=self._max_progress,
                progress_delta=0,
                score_delta=0,
                coin_delta=0,
                time_delta=0,
                player_y_delta=0,
            )
        ), {
            "world": self.mario.world,
            "level": int(self.mario.level),
        }

    def _is_level_complete(self, player_state: int, mode: int) -> bool:
        """Return whether the game has entered a flagpole or castle victory state."""

        return player_state in self.LEVEL_END_STATES or (
            mode == self.WORLD_MAP_MODE and self._previous_mode == self.LEVEL_MODE
        )

    def _update_stagnation(self, new_progress: int) -> None:
        if new_progress >= self.MIN_PROGRESS_ADVANCE:
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        self._episode_steps += 1
        previous_support = self._support_features()[1]
        gap_ahead = not previous_support[1] or not previous_support[2]
        moving_right = action < 4
        jumping = "a" in self.ACTIONS[action]

        for button in self.ACTIONS[action]:
            self.pyboy.button(button, delay=self.frame_skip)
        if self.render_mode == "human":
            for _ in range(self.frame_skip):
                assert self.pyboy.tick(1, True, False)
        else:
            self.pyboy.tick(self.frame_skip, False, False)

        raw_progress = int(self.mario.level_progress)
        progress = raw_progress - self._level_start_progress
        score = int(self.mario.score)
        coins = int(self.mario.coins)
        progress_delta = progress - self._previous_progress
        previous_max_progress = self._max_progress
        new_progress = max(0, progress - self._max_progress)
        self._max_progress = max(self._max_progress, progress)
        score_delta = score - self._previous_score
        coin_delta = coins - self._previous_coins
        time_left = int(self.mario.time_left)
        time_delta = time_left - self._previous_time
        player_y = int(self.pyboy.memory[self.player_y_address])
        player_y_delta = self._signed_byte_delta(player_y, self._previous_player_y)
        player_state = int(self.pyboy.memory[self.PLAYER_STATE_ADDRESS])
        mode = int(self.pyboy.memory[self.MODE_ADDRESS])
        self.route_memory.record_motion(
            progress_delta,
            moving_left=4 <= action < 8,
            in_level=mode == self.LEVEL_MODE,
        )
        self._observe_route(progress)
        progress_reward = 0.05 * new_progress
        backward_penalty = self.BACKWARD_PROGRESS_PENALTY * max(0, -progress_delta)
        score_reward = min(self.SCORE_REWARD_SCALE * max(0, score_delta), self.MAX_SCORE_EVENT_REWARD)
        coin_reward = self.COIN_REWARD_SCALE * max(0, coin_delta)
        gap_approach_penalty = self.GAP_APPROACH_PENALTY if moving_right and gap_ahead and not jumping else 0.0
        gap_jump_reward = self.GAP_JUMP_REWARD if moving_right and gap_ahead and jumping else 0.0
        reward = (
            progress_reward
            - backward_penalty
            + score_reward
            + coin_reward
            - gap_approach_penalty
            + gap_jump_reward
            - 0.01
        )
        if (
            self.curriculum_target is not None
            and previous_max_progress < self.curriculum_target <= self._max_progress
        ):
            reward += 20.0
        level_complete = self._is_level_complete(player_state, mode)
        level_transition_failure = mode == self.OVERWORLD_INIT_MODE
        terminated = bool(self.mario.game_over()) or level_transition_failure
        reached_goal = level_complete
        if reached_goal:
            reward += 100.0
            terminated = True
        elif terminated:
            reward -= 25.0
        stuck = bool(getattr(self.mario, "stuck", False))
        if mode == self.LEVEL_MODE:
            self._update_stagnation(new_progress)
        else:
            self._stagnation_steps = 0
        stagnating = self._stagnation_steps >= self.max_stagnation_steps
        truncated = not terminated and (
            stuck or stagnating or self._episode_steps >= self.max_episode_steps
        )

        self._previous_progress = progress
        self._previous_score = score
        self._previous_coins = coins
        self._previous_time = time_left
        self._previous_player_y = player_y
        self._previous_mode = mode
        features = self._state_features(
            progress=progress,
            max_progress=self._max_progress,
            progress_delta=progress_delta,
            score_delta=score_delta,
            coin_delta=coin_delta,
            time_delta=time_delta,
            player_y_delta=player_y_delta,
        )
        observation = self._observation(features)
        info = {
            "progress": progress,
            "raw_progress": raw_progress,
            "level_start_progress": self._level_start_progress,
            "score": score,
            "coins": coins,
            "time_left": time_left,
            "world": self.mario.world,
            "level": int(self.mario.level),
            "reached_goal": reached_goal,
            "level_complete": level_complete,
            "max_progress": self._max_progress,
            "player_state": player_state,
            "mode": mode,
            "level_transition_failure": level_transition_failure,
            "episode_steps": self._episode_steps,
            "stuck": stuck,
            "stuck_frames": int(getattr(self.mario, "stuck_frames", 0)),
            "stagnating": stagnating,
            "stagnation_steps": self._stagnation_steps,
            "progress_reward": progress_reward,
            "backward_penalty": backward_penalty,
            "score_reward": score_reward,
            "coin_reward": coin_reward,
            "gap_approach_penalty": gap_approach_penalty,
            "gap_jump_reward": gap_jump_reward,
            "route_features": self.route_memory.features(progress),
        }
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray.copy()
        return None

    def close(self) -> None:
        if getattr(self, "pyboy", None) is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
