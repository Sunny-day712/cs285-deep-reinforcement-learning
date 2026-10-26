"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        output_dim = action_dim * chunk_size
        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        prediction = self.network(state)

        prediction = prediction.reshape(
            -1,
            self.chunk_size,
            self.action_dim,
        )

        loss = F.mse_loss(prediction, action_chunk)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        with torch.no_grad():
            actions = self.network(state)

            actions = actions.reshape(
                -1,
                self.chunk_size,
                self.action_dim,
            )

        return actions

class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        input_dim = state_dim + action_dim * chunk_size + 1
        output_dim = action_dim * chunk_size

        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = state[0]

        noise = torch.randn_like(action_chunk)

        t = torch.rand(
            batch_size,1,1,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )

        noisy_actions = t * action_chunk + (1-t) * noise

        target_velocity = action_chunk - noisy_actions

        x = torch.cat(
            [
                state,
                noisy_actions.flatten(start_dim=1),
                t.reshape(batch_size, 1),
            ],
            dim=-1,
        )

        pred_velocity = self.network(x)

        pred_velocity = pred_velocity.reshape(
            batch_size,
            self.action_dim,
            self.chunk_size,
        )

        loss = F.mse_loss(target_velocity, pred_velocity)

        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        batch_size = state[0]

        actions = torch.rand_like(batch_size, self.action_dim, self.chunk_size)

        dt = 1 / num_steps

        for i in range(num_steps):
            t_value = i * dt

            t = torch.full(
                (batch_size, 1),
                t_value,
                device=state.device,
                dtype=state.dtype,
            )

            x = torch.cat(
                [
                    state,
                    actions.flatten(start_dim=1),
                    t,
                ],
                dim=-1,
            )

            velocity = self.network(x)

            velocity = velocity.reshape(
                batch_size,
                self.chunk_size,
                self.action_dim,
            )

            actions = actions + dt * velocity

        return actions

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
