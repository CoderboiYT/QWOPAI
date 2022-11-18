TRAIN = True
METHOD = "IMITATION"
PATH = "./neat-data"

# === Game environment code === #
PORT = 8000
PRESS_DURATION = 0.1
MAX_EPISODE_DURATION_SECS = 10
ACTIONS = {
    0: 'qw',
    1: 'qo',
    2: 'qp',
    3: 'q',
    4: 'wo',
    5: 'wp',
    6: 'w',
    7: 'op',
    8: 'o',
    9: 'p',
    10: '',
}
INIT_X = -3.5

# === Genetic algorithm code === #
GEN_SIZE = 30
GEN_COUNT = 50

NNET_INPUTS = 71
NNET_HIDDEN = 20
NNET_OUTPUTS = 11

MUTATION_WEIGHT_MODIFY_CHANCE = 0.2
MUTATION_ARRAY_MIX_PERC = 0.5
MUTATION_CUTOFF = 0.4
MUTATION_BAD_TO_KEEP = 0.2
MUTATION_MODIFY_CHANCE_LIMIT = 0.4

# === NEAT Code === #
NEAT_CKPT = -1

# === RL Code === #
RL_EPISODES = 1000