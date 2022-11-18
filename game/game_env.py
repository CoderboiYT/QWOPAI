import time

from pynput.keyboard import Controller
from pynput.keyboard import Key
from selenium import webdriver
import numpy as np
import gym
from gym import spaces

from utils import *


class GameEnv(gym.Env):
    meta_data = {'render.modes': ['human']}
    pressed_keys = set()

    def __init__(self):
        super(GameEnv, self).__init__()

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(NNET_INPUTS, ), dtype=np.float32)

        self.gameover = False
        self.time_up = 1
        self.time_down = 1
        self.start_time = time.time()
        self.human_input = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0

        self.driver = webdriver.Chrome()
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')

        time.sleep(2)
        self.driver.find_element_by_xpath("//body").click()

        self.keyboard = Controller()
        self.last_press_time = time.time()

        self.start_x = self.driver.execute_script(
            f'return globalbodystate;')['torso']['position_x']

    def _get_variable_(self, variable):
        return self.driver.execute_script(f'return {variable};')

    def _reset_pressed_keys(self):
        for char in self.pressed_keys:
            self.keyboard.release(char)

        self.pressed_keys.clear()

    # === Internal calculation methods ===
    def _edit_body_state(self, body_state):
        torso_x = body_state["torso"]["position_x"]
        for part, values in body_state.items():
            if 'position_x' in values:
                body_state[part]['position_x'] -= torso_x

        return body_state

    def _calculate_fitness(self, body_state, game_state):
        fitness = 0

        # Reward for distance travelled
        distance = body_state['torso']['position_x'] - self.start_x

        # Punish if going back
        if distance < -3:
            fitness = -10
        # Punish if fell down
        elif game_state['gameEnded'] > 0:
            fitness = distance - 20
        else:
            fitness = distance

            # Punish for time spent close to ground
            # print(body_state['torso']['position_y'])
            if body_state['torso']['position_y'] > 2.0:
                self.time_down += 1
            else:
                self.time_up += 1
            fraction_standing = self.time_up / \
                (self.time_up + self.time_down)

            if fraction_standing < 0.4:
                fitness = -2

            # # Punish for taking more time
            # time_taken = time.time() - self.start_time
            # speed = distance / time_taken

            # fitness = speed * fraction_standing

        return fitness

        return reward

    def _calculate_end_time(self, distance):
        multiplier = (abs(distance) // 15) + 1

        return min(MAX_EPISODE_DURATION_SECS * multiplier, 300)

    def _calculate_game_over(self, game_state, body_state, end_time, fitness):
        torso_x = body_state['torso']['position_x']

        return game_state['gameEnded'] > 0 or \
            game_state['gameOver'] > 0 or \
            game_state['scoreTime'] > end_time or \
            torso_x < -3 or fitness <= -2

    # === Internal interaction methods ===
    def _get_inputs(self):
        game_state = self.driver.execute_script(f'return globalgamestate;')
        body_state = self.driver.execute_script(f'return globalbodystate;')

        fitness = self._calculate_fitness(body_state, game_state)

        end_time = self._calculate_end_time(
            body_state['torso']['position_x'] - self.start_x)

        self.gameover = self._calculate_game_over(
            game_state, body_state, end_time, fitness)

        if self.gameover:
            print(f"============================== Score: {game_state['score']}, Time: {game_state['scoreTime']}")

        body_state = self._edit_body_state(body_state)

        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)

        state = np.array(state, dtype=np.float32).flatten()
        return state, fitness, self.gameover

    def _take_action(self, keys):
        self._reset_pressed_keys()

        for char in keys:
            self.keyboard.press(char)
            self.pressed_keys.add(char)

        time.sleep(PRESS_DURATION)

    def _calculate_keys(actions):
        keys = ""
        if actions[0] > 0.5:
            keys += "q"
        if actions[1] > 0.5:
            keys += "w"
        if actions[2] > 0.5:
            keys += "o"
        if actions[3] > 0.5:
            keys += "p"

        return keys

    def reset(self):
        self._take_action(['r', Key.space])
        self.gameover = False
        self.time_up = 1
        self.time_down = 1
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.start_time = time.time()
        self._reset_pressed_keys()
        self.start_x = self.driver.execute_script(
            f'return globalbodystate;')['torso']['position_x']

        inputs, _, _ = self._get_inputs()
        return inputs

    def step(self, action):
        # keys = GameEnv._calculate_keys(action)

        # print(action)
        # action = np.argmax(action)
        keys = ACTIONS[action]

        if not self.human_input:
            self._take_action(keys)
        else:
            time.sleep(PRESS_DURATION)

        inputs, fitness, done = self._get_inputs()

        inputs = inputs.astype(np.float64)

        return inputs, fitness, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.driver.close()

    def learn(self, net):
        fitness = 0

        inputs, _ = self.reset()

        start_time = time.time()
        while not self.gameover:
            fitness = self.step(net)
        end_time = time.time()

        print(fitness, end_time-start_time)

        return fitness
