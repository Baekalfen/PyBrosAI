"""Demo a recurrent PPO v18 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mario_env import MarioDeluxeEnv
from ppo import ActorCritic, load_checkpoint
from pyboy.utils import WindowEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using greedy actions",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Start and stop PyBoy's GIF screen recorder around the demo",
    )
    return parser.parse_args()


def _screen_recorder(environment: MarioDeluxeEnv):
    return getattr(getattr(environment.pyboy, "_plugin_manager", None), "screen_recorder", None)


def _start_gif_recording(environment: MarioDeluxeEnv) -> None:
    recorder = _screen_recorder(environment)
    if recorder is None or not recorder.enabled():
        raise RuntimeError("PyBoy screen recording is unavailable")
    environment.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
    assert environment.pyboy.tick(1, True, False)


def _stop_gif_recording(environment: MarioDeluxeEnv) -> None:
    environment.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
    assert environment.pyboy.tick(1, True, False)


def run_demo(
    rom: Path,
    checkpoint: Path,
    demo_steps: int,
    *,
    seed: int = 7,
    stochastic: bool = False,
    world_level: tuple[int, int] = (1, 1),
    gif: bool = False,
) -> None:
    frame_skip = 2
    frame_stack = 12
    environment = MarioDeluxeEnv(
        rom,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        render_mode="human",
        world_level=world_level,
    )
    gif_recording = False
    action_names = tuple("+".join(action) if action else "noop" for action in environment.ACTIONS)
    try:
        model = ActorCritic(
            environment.num_tiles,
            frame_stack=frame_stack,
            action_count=len(environment.ACTIONS),
            feature_count=environment.feature_count,
            hidden_size=128,
        )
        load_checkpoint(checkpoint, model, device="cpu")
        model.eval()
        torch.manual_seed(seed)
        observation, _ = environment.reset(seed=seed)
        hidden = torch.zeros((1, 1, model.hidden_size))
        action_counts = [0] * len(action_names)
        max_progress = 0
        if gif:
            _start_gif_recording(environment)
            gif_recording = True

        for _ in range(demo_steps):
            with torch.no_grad():
                distribution, _, hidden = model(
                    torch.as_tensor(observation["tiles"][None], dtype=torch.long),
                    torch.as_tensor(observation["features"][None], dtype=torch.float32),
                    hidden,
                )
            action = int(distribution.sample().item()) if stochastic else int(distribution.probs.argmax().item())
            action_counts[action] += 1
            observation, _, terminated, truncated, info = environment.step(action)
            max_progress = max(max_progress, int(info["max_progress"]))
            if terminated or truncated:
                print(
                    {
                        "steps": sum(action_counts),
                        "max_progress": max_progress,
                        "level_complete": info["level_complete"],
                        "actions": dict(zip(action_names, action_counts, strict=True)),
                    }
                )
                return
        print(
            {
                "steps": demo_steps,
                "max_progress": max_progress,
                "actions": dict(zip(action_names, action_counts, strict=True)),
            }
        )
    finally:
        if gif_recording:
            _stop_gif_recording(environment)
        environment.close()


def main() -> None:
    args = parse_args()
    run_demo(
        args.rom,
        args.checkpoint,
        args.demo_steps,
        seed=args.seed,
        stochastic=args.stochastic,
        world_level=(args.world, args.level),
        gif=args.gif,
    )


if __name__ == "__main__":
    main()
