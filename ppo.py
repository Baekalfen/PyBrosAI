"""Recurrent PPO components for the Mario environment."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Tile-embedding CNN with a GRU policy and value head."""

    def __init__(
        self,
        num_tiles: int,
        frame_stack: int,
        action_count: int,
        feature_count: int,
        hidden_size: int = 128,
        spatial_pool: tuple[int, int] = (8, 10),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        embedding_size = 8
        self.embedding = nn.Embedding(num_tiles, embedding_size)
        self.spatial_pool = spatial_pool
        self.visual = nn.Sequential(
            nn.Conv2d(frame_stack * embedding_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(spatial_pool),
            nn.Flatten(),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(feature_count, 32), nn.Tanh())
        self.recurrent = nn.GRU(64 * spatial_pool[0] * spatial_pool[1] + 32, hidden_size)
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def _encode(self, tiles: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tiles.long())
        leading = embedded.shape[:-4]
        stack, height, width, channels = embedded.shape[-4:]
        visual_input = embedded.permute(*range(len(leading)), -4, -1, -3, -2)
        visual_input = visual_input.reshape(-1, stack * channels, height, width)
        visual = self.visual(visual_input).reshape(*leading, -1)
        feature_input = features.reshape(-1, features.shape[-1])
        encoded_features = self.feature_encoder(feature_input).reshape(*leading, -1)
        return torch.cat((visual, encoded_features), dim=-1)

    def forward(
        self,
        tiles: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        encoded = self._encode(tiles, features)
        sequence = encoded.ndim == 3
        if not sequence:
            encoded = encoded.unsqueeze(0)
        recurrent, hidden = self.recurrent(encoded, hidden)
        logits = self.actor(recurrent)
        values = self.critic(recurrent).squeeze(-1)
        if not sequence:
            logits, values = logits[0], values[0]
        return Categorical(logits=logits), values, hidden

    def evaluate_sequence(
        self,
        tiles: torch.Tensor,
        features: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropies, values = [], [], []
        hidden = initial_hidden
        for step in range(tiles.shape[0]):
            if step:
                hidden = hidden * (1.0 - dones[step - 1]).view(1, -1, 1)
            distribution, value, hidden = self(tiles[step], features[step], hidden)
            log_probs.append(distribution.log_prob(actions[step]))
            entropies.append(distribution.entropy())
            values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


@dataclass
class Rollout:
    tile_observations: torch.Tensor
    feature_observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_hidden: torch.Tensor


def collect_rollout(
    model: ActorCritic,
    environments: Iterable,
    observations: list[dict[str, np.ndarray]],
    steps: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> tuple[Rollout, list[dict[str, np.ndarray]], list[dict[str, float]]]:
    environments = list(environments)
    count = len(environments)
    tile_buffer, feature_buffer = [], []
    action_buffer, log_prob_buffer = [], []
    reward_buffer, done_buffer, value_buffer = [], [], []
    episode_returns = [0.0] * count
    episode_lengths = [0] * count
    episode_max_progress = [0] * count
    episode_score = [0] * count
    episode_coins = [0] * count
    completed_episodes: list[dict[str, float]] = []
    hidden = torch.zeros((1, count, model.hidden_size), device=device)
    initial_hidden = hidden.detach().clone()

    def advance(environment, action):
        next_observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        if done:
            next_observation, _ = environment.reset()
        return next_observation, reward, done, info

    rollout_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    stream_context = torch.cuda.stream(rollout_stream) if rollout_stream is not None else nullcontext()
    with torch.no_grad(), ThreadPoolExecutor(max_workers=count) as executor, stream_context:
        for _ in range(steps):
            tiles = torch.as_tensor(
                np.stack([observation["tiles"] for observation in observations]), dtype=torch.long, device=device
            )
            features = torch.as_tensor(
                np.stack([observation["features"] for observation in observations]),
                dtype=torch.float32,
                device=device,
            )
            distribution, values, hidden = model(tiles, features, hidden)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            transitions = list(executor.map(advance, environments, actions.tolist()))

            next_observations, rewards, dones = [], [], []
            for index, (next_observation, reward, done, info) in enumerate(transitions):
                next_observations.append(next_observation)
                rewards.append(reward)
                dones.append(float(done))
                episode_returns[index] += reward
                episode_lengths[index] += 1
                episode_max_progress[index] = max(
                    episode_max_progress[index], int(info.get("max_progress", info.get("progress", 0)))
                )
                episode_score[index] = max(episode_score[index], int(info.get("score", 0)))
                episode_coins[index] = max(episode_coins[index], int(info.get("coins", 0)))
                if done:
                    completed_episodes.append(
                        {
                            "return": episode_returns[index],
                            "length": float(episode_lengths[index]),
                            "max_progress": float(episode_max_progress[index]),
                            "score": float(episode_score[index]),
                            "coins": float(episode_coins[index]),
                            "success": float(info.get("level_complete", False)),
                        }
                    )
                    episode_returns[index] = 0.0
                    episode_lengths[index] = 0
                    episode_max_progress[index] = 0
                    episode_score[index] = 0
                    episode_coins[index] = 0
            done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
            hidden = hidden * (1.0 - done_tensor).view(1, -1, 1)

            tile_buffer.append(tiles)
            feature_buffer.append(features)
            action_buffer.append(actions)
            log_prob_buffer.append(log_probs)
            reward_buffer.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
            done_buffer.append(done_tensor)
            value_buffer.append(values)
            observations = next_observations

        next_tiles = torch.as_tensor(
            np.stack([observation["tiles"] for observation in observations]), dtype=torch.long, device=device
        )
        next_features = torch.as_tensor(
            np.stack([observation["features"] for observation in observations]), dtype=torch.float32, device=device
        )
        _, next_values, _ = model(next_tiles, next_features, hidden)
    if rollout_stream is not None:
        rollout_stream.synchronize()

    rewards = torch.stack(reward_buffer)
    dones = torch.stack(done_buffer)
    values = torch.stack(value_buffer)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(count, dtype=torch.float32, device=device)
    for index in reversed(range(steps)):
        next_non_terminal = 1.0 - dones[index]
        next_value = next_values if index == steps - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage

    return (
        Rollout(
            tile_observations=torch.stack(tile_buffer),
            feature_observations=torch.stack(feature_buffer),
            actions=torch.stack(action_buffer),
            log_probs=torch.stack(log_prob_buffer),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=advantages + values,
            initial_hidden=initial_hidden,
        ),
        observations,
        completed_episodes,
    )


def update_policy(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> dict[str, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (rollout.advantages.std() + 1e-8)
    environment_count = rollout.actions.shape[1]
    sequence_minibatch = max(1, min(environment_count, minibatch_size // rollout.actions.shape[0]))
    metrics: dict[str, float] = {}
    for _ in range(epochs):
        indices = np.random.permutation(environment_count)
        for start in range(0, environment_count, sequence_minibatch):
            selected = torch.as_tensor(indices[start : start + sequence_minibatch], device=rollout.actions.device)
            log_probs, entropy, values = model.evaluate_sequence(
                rollout.tile_observations[:, selected],
                rollout.feature_observations[:, selected],
                rollout.dones[:, selected],
                rollout.actions[:, selected],
                rollout.initial_hidden[:, selected],
            )
            ratio = (log_probs - rollout.log_probs[:, selected]).exp()
            selected_advantages = advantages[:, selected]
            policy_loss = -torch.minimum(
                ratio * selected_advantages,
                ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * selected_advantages,
            ).mean()
            value_loss = 0.5 * (rollout.returns[:, selected] - values).square().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.mean().detach()),
            }
    return metrics


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_return: float | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_return": best_return},
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint["model"]
    embedding = state.get("embedding.weight")
    if embedding is not None and embedding.shape != model.embedding.weight.shape:
        raise RuntimeError(
            f"Checkpoint uses {embedding.shape[0]} tile IDs, but this environment uses "
            f"{model.embedding.weight.shape[0]}. Start a new run with a fresh checkpoint."
        )
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))
