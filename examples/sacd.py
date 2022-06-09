# pyright: reportPrivateImportUsage=false
import gym
import optax
from absl import app, flags
from trust.agents import SACD, SacHParams
from trust.environments.downhill import MinihackToDMEnv
from trust import experiment
from flax import linen as nn
from helx.image import imresize


FLAGS = flags.FLAGS

# --- Experiment
flags.DEFINE_bool(
    "debug",
    False,
    "Use this flag to run the experiment in debug mode",
)
flags.DEFINE_integer(
    "num_episodes",
    1_000_000,
    "Number of training episodes to run",
)
flags.DEFINE_string(
    "env",
    "MiniHack-Downhill-5x5-v0",
    "The name of the gym environment to experiment on",
)
flags.DEFINE_integer(
    "seed",
    0,
    "Random seed to control the experiment",
)

# --- RL
flags.DEFINE_integer(
    "n_steps",
    1,
    "Number of timesteps for multistep return",
)
flags.DEFINE_float(
    "discount",
    1.0,
    "Discount factor gamma used in the Q-learning update",
)
flags.DEFINE_float(
    "tau",
    5e-3,
    "Target smoothing coefficient for polyak interpolation",
)

# --- Deep RL
flags.DEFINE_integer(
    "replay_start",
    1000,
    "A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory",
)
flags.DEFINE_integer(
    "replay_memory_size",
    3000,
    "SGD updates are sampled from this number of most recent frames",
)
flags.DEFINE_integer(
    "update_frequency",
    1,
    "The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates.",
)
flags.DEFINE_integer(
    "target_network_update_frequency",
    200,
    "The frequency (measured in the number of parameters update) with which the target network is updated (this corresponds to the parameter C from Algorithm 1)",
)

# --- NN
flags.DEFINE_integer(
    "hidden_size",
    128,
    "Dimension of last linear layer for value regression",
)

# --- SGD
flags.DEFINE_integer(
    "batch_size",
    32,
    "Number of training cases over which each stochastic gradient descent (SGD) update is computed",
)
flags.DEFINE_float(
    "learning_rate",
    3e-4,
    "The learning rate used by the RMSProp",
)


def main(argv):
    SEED = FLAGS.seed
    HIDDEN_SIZE = FLAGS.hidden_size
    NUM_EPISODES = FLAGS.num_episodes
    ENV = FLAGS.env
    DEBUG = FLAGS.debug

    class Encoder(nn.Module):
        n_out: int

        @nn.compact
        def __call__(self, x):
            x = imresize(x, (56, 56), channel_first=False)
            x = nn.Conv(8, (8, 8), (4, 4), "VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(16, (4, 4), (2, 2), "VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(16, (3, 3), (1, 1), "VALID")(x)
            x = nn.relu(x)
            x = x.reshape(1, -1)
            x = nn.Dense(HIDDEN_SIZE)(x)
            x = nn.relu(x)
            x = nn.Dense(self.n_out)(x)
            return x

    class Actor(nn.Module):
        n_actions: int

        @nn.compact
        def __call__(self, x):
            return Encoder(self.n_actions)(x)

    class Critic(nn.Module):
        n_actions: int

        @nn.compact
        def __call__(self, x):
            # twinned network
            q_A = Encoder(self.n_actions)(x)
            q_B = Encoder(self.n_actions)(x)
            return q_A, q_B

    # setup env
    env = gym.make(ENV)
    env = MinihackToDMEnv(env)

    # setup agent
    hp = SacHParams(
        input_shape=env.observation_spec().shape,
        tau=FLAGS.tau,
        replay_start=FLAGS.replay_start,
        replay_memory_size=FLAGS.replay_memory_size,
        update_frequency=FLAGS.update_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        discount=FLAGS.discount,
        n_steps=FLAGS.n_steps,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
    )
    if DEBUG:
        hp = SacHParams(
            input_shape=env.observation_spec().shape,
            tau=FLAGS.tau,
            replay_start=32,
            replay_memory_size=FLAGS.replay_memory_size,
            update_frequency=1,
            target_network_update_frequency=1,
            discount=FLAGS.discount,
            n_steps=FLAGS.n_steps,
            batch_size=2,
            learning_rate=FLAGS.learning_rate,
        )

    num_actions = env.action_spec().num_values
    actor = Actor(num_actions)
    critic = Critic(num_actions)
    optimiser = optax.adam(learning_rate=hp.learning_rate)
    agent = SACD(actor, critic, optimiser, hp, SEED)

    # run experiment
    experiment.run(
        agent=agent,
        env=env,
        num_episodes=NUM_EPISODES,
        project="example",
        experiment_name="example",
        debug=DEBUG,
    )


if __name__ == "__main__":
    app.run(main)
