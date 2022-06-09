from __future__ import annotations

from typing import Tuple
from chex import Array, PRNGKey
import dm_env
import gym
from bsuite.utils.gym_wrapper import space2spec
from dm_env.specs import DiscreteArray
from gym import spaces
from minihack import MiniHack
from minihack.envs.room import MiniHackRoom
from nle import nethack


ACTION_SET = (nethack.CompassDirection(106), nethack.CompassDirection(108))
OBSERVATION_KEY = "pixel_crop"


class MinihackToDMEnv(dm_env.Environment):
    """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

    def __init__(self, gym_env: MiniHack):
        self.minihack_env = gym_env
        self.name = self.minihack_env.unwrapped.spec.name  # type: ignore
        # Convert gym observation space to dm_env specs.
        self._observation_spec = space2spec(
            self.minihack_env.observation_space, name="observations"
        )
        # Convert gym action space to dm_env specs.
        assert isinstance(gym_env.action_space, spaces.Discrete)
        self._action_spec = DiscreteArray(
            gym_env.action_space.n, dtype=gym_env.action_space.dtype, name="actions"
        )
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation = self.minihack_env.reset()[OBSERVATION_KEY]  # type: ignore
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        # Convert the gym step result to a dm_env TimeStep.
        observation, reward, done, info = self.minihack_env.step(action)
        observation = observation[OBSERVATION_KEY]
        self._reset_next_step = done

        if done:
            is_truncated = (
                info.get("end_status") is self.minihack_env.StepStatus.ABORTED
            )
            if is_truncated:
                return dm_env.truncation(reward, observation)
            else:
                return dm_env.termination(reward, observation)
        else:
            return dm_env.transition(reward, observation)

    def close(self):
        self.minihack_env.close()

    def observation_spec(self):
        return self._observation_spec[OBSERVATION_KEY]  # type: ignore

    def action_spec(self):
        return self._action_spec

    def render(self, mode="human"):
        return self.minihack_env.render(mode)


class MiniHackDownhill(MiniHackRoom):
    def __init__(self, size, *args, **kwargs):
        super().__init__(
            *args,
            size=size,
            random=False,
            actions=ACTION_SET,
            penalty_step=0.0,
            max_episode_steps=size * 2,
            observation_keys=(OBSERVATION_KEY,),
            **kwargs,
        )

    def oracle(
        self, observation: Array, key: PRNGKey | None = None, eval: bool = False
    ) -> Tuple[Array, Array]:
        """An oracle policy for the optimal credit problem.
        This is not only an optimal policy, but the optimal policy
        with the highest entropy."""
        return

    def step(self, action: int):
        #  we override `self.step` to make sure that
        #  the agent does not get stuck onto a wall
        frozen_steps = self._frozen_steps
        new_timestep = super().step(action)
        is_frozen = self._frozen_steps > frozen_steps
        if is_frozen:
            # cancel current step
            self._steps -= 1
            # action is binary (0, 1), so we take the other one
            new_timestep = super().step(not action)
        return new_timestep


class MiniHackDownhill5x5(MiniHackDownhill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=5, **kwargs)


class MiniHackDownhill15x15(MiniHackDownhill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=15, **kwargs)


gym.register(
    id="MiniHack-Downhill-5x5-v0",
    entry_point="experiments.envs.downhill:MiniHackDownhill5x5",
)
gym.register(
    id="MiniHack-Downhill-15x15-v0",
    entry_point="experiments.envs.downhill:MiniHackDownhill15x15",
)
