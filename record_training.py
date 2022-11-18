from stable_baselines.gail import generate_expert_traj

from game.game_env import GameEnv

from utils import ACTIONS

MAPPING = {
    'qw': 0,
    'qo': 1,
    'qp': 2,
    'q': 3,
    'wo': 4,
    'wp': 5,
    'w': 6,
    'op': 7,
    'o': 8,
    'p': 9,
    '': 10,
}


def generator(_):

    env.human_input = True
    game_state = env._get_variable_('globalgamestate')

    string = ''
    for char in ['q', 'w', 'o', 'p']:
        if game_state[char]:
            string = string + char
    if 'q' in string and 'p' in string:
        string = 'qp'
    if 'w' in string and 'o' in string:
        string = 'wo'
    string = string[:2]
    for _, value in ACTIONS.items():
        if set(string) == set(value):
            return value

    raise ValueError(f'Key presses not found {string}')
 

env = GameEnv()
generate_expert_traj(generator, "./expert", env, n_episodes=10)