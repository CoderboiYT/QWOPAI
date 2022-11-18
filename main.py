import os
import neat
import time
import pickle

from game.game_env import GameEnv
from nnet import NNet
from utils import *

env = GameEnv()

# === Genetic algorithm code === #
def fetch_model(model_path):
    model = NNet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)

    if os.path.isfile(model_path + '/input_hidden.txt'):
        model.load_weights(model_path)

    return model


def train_genetic(path):
    models = [fetch_model(path) for _ in range(GEN_SIZE)]

    for gen in range(GEN_COUNT):
        fitness_sum = 0
        for i, model in enumerate(models):
            fitness = genetic_stepper(model)
            model.fitness = fitness
            fitness_sum += model.fitness

        models = NNet.evolve_population(models)

        print(f"Generation {gen} complete, fitness = {fitness_sum/GEN_SIZE}")
        model.save_weights()

    print("Finished training model")


def genetic_stepper(net):
    fitness = 0

    inputs = env.reset()

    start_time = time.time()
    while not env.gameover:
        action = net.activate(inputs)
        inputs, fitness, _, info = env.step(action)
    end_time = time.time()

    return fitness

# === NEAT Code === #
def eval_genomes(genomes, config):

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = neat_stepper(net)


def train_neat(config):
    if NEAT_CKPT != -1:
        p = neat.Checkpointer.restore_checkpoint(
            f"neat-checkpoint-{NEAT_CKPT}")
    else:
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def neat_stepper(net):
    fitness = 0

    inputs = env.reset()

    start_time = time.time()
    while not env.gameover:
        action = net.activate(inputs)
        inputs, fitness, _, info = env.step(action)
    end_time = time.time()

    print(fitness, end_time-start_time)

    return fitness


def load_config():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "./neat-config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    return config

# === RL Code === #
def train_rl():
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    # from stable_baselines3 import A2C
    # model = A2C("MlpPolicy", env, verbose=1)

    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # from stable_baselines import DQN
    # model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

    # from stable_baselines.common.policies import MlpPolicy
    # from stable_baselines import ACER

    # model = ACER(MlpPolicy, env, verbose=1)

    print("===== Imported =====")

    model.learn(total_timesteps=1_100_000)

    # for ep in range(RL_EPISODES):
    #     obs = env.reset()
    #     done = False

    #     while not done:
    #         env.render()
    #         action, _states = model.predict(obs)
    #         obs, reward, done, info = env.step(action)

def train_imitation():
    from stable_baselines import GAIL
    from stable_baselines.gail import ExpertDataset

    dataset = ExpertDataset(expert_path='expert.npz', traj_limitation=-1, verbose=1)

    model = GAIL('MlpPolicy', env, dataset, verbose=1)
    # model = GAIL.load("gail_qwop_110", env=env)

    # Using for loop to save intermittent models
    for i in range(114, 240):
        # Note: in practice, you need to train for 1M steps to have a working policy
        model.learn(total_timesteps=5_000)
        model.save(f"gail_qwop_{i}")
        print(f"=========== {i} 5,000 TIMESTEMPS DONE ===========")

def run_test():
    # Initialize env and model
    from stable_baselines import GAIL

    model = GAIL.load("gail_qwop_114")

    for _ in range(2_000):
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

        time.sleep(1)

def main():
    if METHOD == "GENETIC":
        train_genetic(PATH)
    elif METHOD == "NEAT":
        train_neat(load_config())
    elif METHOD == "RL":
        train_rl()
    elif METHOD == "IMITATION":
        train_imitation()
    else:
        run_test()


if __name__ == "__main__":
    main()
