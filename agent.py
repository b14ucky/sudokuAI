import torch
import random
import pyautogui
from collections import deque
from game_player import GamePlayer
from models import SudokuSolverModel, SudokuSolverTrainer
from board_recognition import BoardRecognizer
from helpers import MouseListener, convert_to_tensor
import matplotlib.pyplot as plt
from IPython import display
from time import sleep


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = SudokuSolverModel()
        self.board_recognizer = BoardRecognizer()
        self.trainer = SudokuSolverTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self):
        return self.board_recognizer.recognize_board()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 100 - self.n_games

        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 728)
        else:
            prediction = self.model(convert_to_tensor(state))
            action = torch.argmax(prediction).item()

        return action


def train():
    record = 3
    last_action = 0
    agent = Agent()
    mouse_listener = MouseListener()
    board_recognizer = BoardRecognizer()

    mouse_listener.wait_for_click()
    if mouse_listener.mouse_clicked:
        x, y = pyautogui.position()
        print(x, y)
    board = board_recognizer.recognize_board(x, y)
    game_player = GamePlayer((x, y), board)
    while True:
        state_old = game_player.board.copy()

        action = agent.get_action(state_old)

        value = action % 9 + 1
        row = action // 81
        column = (action - row * 81) // 9
        reward, done, mistakes = game_player.insert_value(column, row, value)
        state_new = game_player.board.copy()
        if reward < 0:
            game_player.board = state_old.copy()

        if last_action == action:
            reward = -30
        last_action = action

        agent.train_short_memory(state_old, action, reward, state_new, done)

        agent.remember(state_old, action, reward, state_new, done)

        if done:
            sleep(0.5)
            game_player.restart_game()
            game_player.board = board_recognizer.recognize_board(x, y)
            game_player.in_game_mistakes = 0
            agent.n_games += 1
            agent.train_long_memory()

            if mistakes < record:
                record = mistakes
                agent.trainer.save()

            print(
                f"Game {agent.n_games} finished with {mistakes} mistakes. Record: {record}"
            )


if __name__ == "__main__":
    train()
