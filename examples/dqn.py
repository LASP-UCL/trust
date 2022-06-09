# pyright: reportPrivateImportUsage=false
import gym
import optax
from absl import app, flags
from trust.agents.dqn import DQN, DqnHParams
from trust.environments.downhill import MinihackToDMEnv
from trust import experiment
from flax import linen as nn
from helx.image import imresize
from helx.random import PRNGSequence


FLAGS = flags.FLAGS

# --- Experiment
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
    "initial_exploration",
    1.0,
    "Initial value of ɛ in ɛ-greedy exploration",
)
flags.DEFINE_float(
    "final_exploration",
    0.01,
    "Final value of ɛ in ɛ-greedy exploration",
)
flags.DEFINE_integer(
    "final_exploration_frame",
    10_000,
    "The number of frames over which the initial value of ɛ is linearly annealed to its final value",
)
flags.DEFINE_integer(
    "no_op_max",
    30,
    'Maximum numer of "do nothing" actions to be performed by the agent at the start of an episode',
)
flags.DEFINE_integer(
    "action_repeat",
    1,
    "Repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4th input frame",
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
flags.DEFINE_integer(
    "agent_history_length",
    1,
    "The number of most recent frames experienced by the agent that are given as  input to the Q network",
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
    0.00025,
    "The learning rate used by the RMSProp",
)
flags.DEFINE_float(
    "gradient_momentum",
    0.95,
    "Gradient momentum used by the RMSProp",
)
flags.DEFINE_float(
    "squared_gradient_momentum",
    0.95,
    "Squared gradient (denominator) momentum used by the RMSProp",
)
flags.DEFINE_float(
    "min_squared_gradient",
    0.01,
    "Constant added to the squared gradient in the denominator of the RMSProp update",
)


def main(argv):
    SEED = FLAGS.seed
    HIDDEN_SIZE = FLAGS.hidden_size
    NUM_EPISODES = FLAGS.num_episodes
    ENV = FLAGS.env

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

    # setup env
    env = gym.make(ENV)
    env = MinihackToDMEnv(env)

    # setup agent
    hp = DqnHParams(
        input_shape=env.observation_spec().shape,
        initial_exploration=FLAGS.initial_exploration,
        final_exploration=FLAGS.final_exploration,
        final_exploration_frame=FLAGS.final_exploration_frame,
        replay_start=FLAGS.replay_start,
        replay_memory_size=FLAGS.replay_memory_size,
        update_frequency=FLAGS.update_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        discount=FLAGS.discount,
        n_steps=FLAGS.n_steps,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        gradient_momentum=FLAGS.gradient_momentum,
        squared_gradient_momentum=FLAGS.squared_gradient_momentum,
        min_squared_gradient=FLAGS.min_squared_gradient,
    )

    network = Encoder(env.action_spec().num_values)
    optimiser = optax.rmsprop(
        learning_rate=hp.learning_rate,
        decay=hp.squared_gradient_momentum,
        eps=hp.min_squared_gradient,
        momentum=hp.gradient_momentum,
    )
    agent = DQN(network, optimiser, hp, SEED)

    # run experiment
    experiment.run(
        agent,
        env,
        NUM_EPISODES,
        project="example",
        experiment_name="example",
        debug=True
    )


if __name__ == "__main__":
    app.run(main)
