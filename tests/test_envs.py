from trust.environments.downhill import MinihackToDMEnv
import gym


def downhill_test():
    ENV = "MiniHack-Downhill-5x5-v0"
    env = gym.make(ENV)
    env = MinihackToDMEnv(env)

    env.reset()

    env.step(0)


if __name__ == "__main__":
    downhill_test()
