"""Train recurrent PPO on Super Mario Bros. Deluxe."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from mario_env import MarioDeluxeEnv
from ppo import ActorCritic, collect_rollout, save_checkpoint, update_policy


def evaluate_policy(
    model: ActorCritic,
    environments: list[MarioDeluxeEnv],
    max_steps: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    stochastic: bool = False,
) -> tuple[float, float, list[tuple[tuple[int, int], float, float]]]:
    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    model.eval()
    try:
        if stochastic:
            torch.manual_seed(seed)
        progress_values = []
        successes = []
        level_metrics = []
        for level_index, environment in enumerate(environments):
            level_progress_values = []
            level_successes = []
            for episode in range(episodes):
                observation, _ = environment.reset(seed=seed + level_index * episodes + episode)
                hidden = torch.zeros((1, 1, model.hidden_size), device=device)
                max_progress = 0
                completed = False
                for _ in range(max_steps):
                    with torch.no_grad():
                        distribution, _, hidden = model(
                            torch.as_tensor(observation["tiles"][None], dtype=torch.long, device=device),
                            torch.as_tensor(observation["features"][None], dtype=torch.float32, device=device),
                            hidden,
                        )
                    action = (
                        int(distribution.sample().item())
                        if stochastic
                        else int(distribution.probs.argmax().item())
                    )
                    observation, _, terminated, truncated, info = environment.step(action)
                    max_progress = max(max_progress, int(info.get("max_progress", info.get("progress", 0))))
                    if terminated or truncated:
                        completed = bool(info.get("level_complete", False))
                        break
                progress_values.append(max_progress)
                successes.append(float(completed))
                level_progress_values.append(max_progress)
                level_successes.append(float(completed))
            level_metrics.append(
                (
                    environment.world_level,
                    float(np.mean(level_progress_values)),
                    float(np.mean(level_successes)),
                )
            )
        return float(np.mean(progress_values)), float(np.mean(successes)), level_metrics
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        model.train(was_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoints/mario_deluxe_recurrent_ppo_v18.pt")
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-steps", type=int, default=1_000_000_000)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--frame-stack", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=6_000)
    parser.add_argument("--max-stagnation-steps", type=int, default=2_400)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument(
        "--levels",
        default="1-1,1-2,1-3,2-1,2-3,3-1,3-2,3-3,4-1,4-2,4-3",
        help="Comma-separated world-levels to evaluate",
    )
    parser.add_argument(
        "--training-levels",
        default="1-1,1-2,1-2,1-3,1-3,2-1,2-3,3-1,3-2,3-3,4-1,4-2,4-3",
        help="Optional weighted comma-separated world-levels for training workers; --levels remains the evaluation set",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--evaluation-interval", type=int, default=50_000)
    parser.add_argument("--evaluation-steps", type=int, default=6_000)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--stochastic-evaluation-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--curriculum-targets", default="1000,1500,2000,2500")
    parser.add_argument("--curriculum-stage-steps", type=int, default=2_000_000)
    args = parser.parse_args()
    args.curriculum_targets = tuple(int(target) for target in args.curriculum_targets.split(",") if target)
    if not args.curriculum_targets:
        raise ValueError("--curriculum-targets must contain at least one progress target")
    if args.curriculum_stage_steps < 1:
        raise ValueError("--curriculum-stage-steps must be positive")
    def parse_levels(value: str, argument_name: str) -> tuple[tuple[int, int], ...]:
        try:
            levels = tuple(
                (int(world), int(level))
                for world, level in (item.split("-", 1) for item in value.split(",") if item)
            )
        except ValueError as error:
            raise ValueError(
                f"{argument_name} must use comma-separated WORLD-LEVEL values such as 1-1,1-2"
            ) from error
        if not levels or any(world < 1 or level < 1 for world, level in levels):
            raise ValueError(f"{argument_name} must contain at least one positive WORLD-LEVEL value")
        return levels

    args.levels = parse_levels(args.levels, "--levels")
    args.training_levels = parse_levels(args.training_levels, "--training-levels") if args.training_levels else args.levels
    if args.best_checkpoint is None:
        checkpoint = Path(args.checkpoint)
        args.best_checkpoint = str(checkpoint.with_name(f"{checkpoint.stem}.best{checkpoint.suffix}"))
    if args.stochastic_evaluation_episodes < 1:
        raise ValueError("--stochastic-evaluation-episodes must be positive")
    args.stochastic_best_checkpoint = str(
        Path(args.checkpoint).with_name(
            f"{Path(args.checkpoint).stem}.stochastic.best{Path(args.checkpoint).suffix}"
        )
    )
    return args


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    environments = [
        MarioDeluxeEnv(
            args.rom,
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            max_stagnation_steps=args.max_stagnation_steps,
            world_level=args.training_levels[index % len(args.training_levels)],
            curriculum_target=args.curriculum_targets[0],
        )
        for index in range(args.num_envs)
    ]
    evaluation_environments = [
        MarioDeluxeEnv(
            args.rom,
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            max_stagnation_steps=args.max_stagnation_steps,
            world_level=world_level,
            curriculum_target=None,
        )
        for world_level in args.levels
    ]
    model = ActorCritic(
        environments[0].num_tiles,
        args.frame_stack,
        environments[0].action_space.n,
        environments[0].feature_count,
        args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    start_step = 0
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint.get("step", 0) >= args.total_steps:
            print(f"Checkpoint already reached target: {checkpoint['step']:,} >= {args.total_steps:,} steps")
            for environment in (*environments, *evaluation_environments):
                environment.close()
            return
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"Resumed checkpoint at {start_step:,} steps")
    print(
        {
            "device": str(device),
            "algorithm": "recurrent_ppo",
            "frame_stack": args.frame_stack,
            "frame_skip": args.frame_skip,
            "frame_shape": (environments[0].height, environments[0].width),
            "action_count": environments[0].action_space.n,
            "checkpoint": args.checkpoint,
        }
    )
    observations = [environment.reset(seed=args.seed + index)[0] for index, environment in enumerate(environments)]
    best_evaluation = (-1, -1)
    best_stochastic_progress = -1.0
    next_log_step = start_step + args.log_interval
    next_evaluation_step = start_step + args.evaluation_interval
    next_checkpoint_step = start_step + args.checkpoint_interval
    try:
        step = start_step
        while step < args.total_steps:
            curriculum_index = min(
                len(args.curriculum_targets) - 1,
                step // args.curriculum_stage_steps,
            )
            for environment in environments:
                environment.curriculum_target = args.curriculum_targets[curriculum_index]
            rollout, observations, completed = collect_rollout(
                model,
                environments,
                observations,
                args.rollout_steps,
                device,
                args.gamma,
                args.gae_lambda,
            )
            completed_steps = len(rollout.actions) * rollout.actions.shape[1]
            step += completed_steps
            metrics = update_policy(
                model,
                optimizer,
                rollout,
                epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
            )
            if step >= next_log_step:
                fields = f"steps={step:,} episodes={len(completed)} {metrics}"
                if completed:
                    fields += (
                        f" progress={np.mean([episode['max_progress'] for episode in completed]):.0f}"
                        f" success={np.mean([episode['success'] for episode in completed]):.0%}"
                        f" return={np.mean([episode['return'] for episode in completed]):.2f}"
                        f" score={np.mean([episode['score'] for episode in completed]):.0f}"
                        f" coins={np.mean([episode['coins'] for episode in completed]):.1f}"
                    )
                if step >= next_evaluation_step:
                    greedy_progress, greedy_success_rate, greedy_level_metrics = evaluate_policy(
                        model,
                        evaluation_environments,
                        args.evaluation_steps,
                        args.seed,
                        args.evaluation_episodes,
                        device,
                    )
                    stochastic_progress, stochastic_success_rate, stochastic_level_metrics = evaluate_policy(
                        model,
                        evaluation_environments,
                        args.evaluation_steps,
                        args.seed,
                        args.stochastic_evaluation_episodes,
                        device,
                        stochastic=True,
                    )
                    score = (greedy_success_rate, greedy_progress)
                    fields += (
                        f" eval_greedy_progress={greedy_progress:.0f}"
                        f" eval_greedy_success={greedy_success_rate:.0%}"
                    )
                    fields += " eval_greedy_levels=" + ",".join(
                        f"{world}-{level}:{level_progress:.0f}/{level_success:.0%}"
                        for (world, level), level_progress, level_success in greedy_level_metrics
                    )
                    if score > best_evaluation:
                        best_evaluation = score
                        save_checkpoint(args.best_checkpoint, model, optimizer, step, float(greedy_progress))
                        fields += " best_greedy"
                    fields += (
                        f" eval_stochastic_progress={stochastic_progress:.0f}"
                        f" eval_stochastic_success={stochastic_success_rate:.0%}"
                    )
                    fields += " eval_stochastic_levels=" + ",".join(
                        f"{world}-{level}:{level_progress:.0f}/{level_success:.0%}"
                        for (world, level), level_progress, level_success in stochastic_level_metrics
                    )
                    normalized_stochastic_progress = sum(
                        min(level_progress, 3300.0) / 3300.0
                        for _, level_progress, _ in stochastic_level_metrics
                    )
                    if normalized_stochastic_progress > best_stochastic_progress:
                        best_stochastic_progress = normalized_stochastic_progress
                        save_checkpoint(
                            args.stochastic_best_checkpoint,
                            model,
                            optimizer,
                            step,
                            float(stochastic_progress),
                        )
                        fields += " best_stochastic_progress"
                    next_evaluation_step += args.evaluation_interval
                print(fields)
                next_log_step += args.log_interval
            if step >= next_checkpoint_step:
                save_checkpoint(args.checkpoint, model, optimizer, step)
                next_checkpoint_step += args.checkpoint_interval
    finally:
        for environment in evaluation_environments:
            environment.close()
        for environment in environments:
            environment.close()


if __name__ == "__main__":
    train(parse_args())
