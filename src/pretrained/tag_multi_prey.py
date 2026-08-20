import gym
from gym.spaces import Tuple
from pretrained.ddpg import DDPG
import torch
import os


class PretrainedTagMultiPrey(gym.Wrapper):
    """Tag wrapper with multiple pretrained prey agents."""

    def __init__(self, *args, n_preys=1, **kwargs):
        super().__init__(*args, **kwargs)
        if n_preys < 1:
            raise ValueError(f"n_preys must be >= 1, got {n_preys}")
        if n_preys >= len(self.action_space):
            raise ValueError(
                f"n_preys={n_preys} is invalid for env with {len(self.action_space)} agents"
            )

        self.n_preys = int(n_preys)
        self.pt_action_space = tuple(self.action_space[-self.n_preys :])
        self.pt_observation_space = tuple(self.observation_space[-self.n_preys :])
        self.action_space = Tuple(self.action_space[: -self.n_preys])
        self.observation_space = Tuple(self.observation_space[: -self.n_preys])
        self.n_agents = len(self.action_space)

        self.prey = DDPG(14, 5, 50, 128, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), "prey_params.pt")
        save_dict = torch.load(param_path, weights_only=True)
        self.prey.load_params(save_dict["agent_params"][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = list(obs[-self.n_preys :])
        return obs[: -self.n_preys]

    def step(self, action):
        prey_actions = tuple(self.prey.step(prey_obs) for prey_obs in self.last_prey_obs)
        action = tuple(action) + prey_actions
        obs, rew, done, info = super().step(action)
        self.last_prey_obs = list(obs[-self.n_preys :])
        obs = obs[: -self.n_preys]
        rew = rew[: -self.n_preys]
        done = done[: -self.n_preys]
        return obs, rew, done, info
