import torch
import pandas as pd
import numpy as np
from pynput import mouse
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class SudokuDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board = self.data.iat[idx, 0]
        board = torch.tensor(list(map(float, board)), dtype=torch.float32).reshape((9, 9)) / 9 - 0.5
        solution = self.data.iat[idx, 1]
        solution = torch.tensor(list(map(float, solution)), dtype=torch.long).reshape((81,)) - 1

        return board, solution


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.genfromtxt(
            file_path,
            delimiter=";",
            dtype=None,
            names=["Labels", "Pixels"],
            skip_header=1,
            encoding=None,
        )
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = int(self.data[idx]["Labels"])
        pixels = np.array(list(map(int, self.data[idx]["Pixels"].split())))
        pixels = pixels.reshape((28, 28)).astype(np.uint8)

        if self.transform:
            pixels = self.transform(pixels)

        return pixels, label


class MouseListener:
    def __init__(self):
        self.mouse_clicked = False
        self.listener = mouse.Listener(on_click=self.on_click)

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            self.mouse_clicked = True
            self.listener.stop()

    def wait_for_click(self):
        print(
            "Align your cursor with the top left corner of the board\nand click the left button to capture the board..."
        )
        self.listener.start()
        self.listener.join()


def display_board(board):
    fig, ax = plt.subplots()
    ax.axis("off")

    plt.title('Check if the board is correct and press "y" to continue.', pad=20)

    display_array = np.where(np.isin(board, 0), None, board)

    table = ax.table(
        cellText=display_array,
        loc="center",
        cellLoc="center",
        colWidths=[0.1] * len(board[0]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for i in range(len(display_array[0])):
        for j in range(len(display_array[0])):
            table[(i, j)].set_height(0.125)

    plt.show()


def normalize(board):
    return board / 9 - 0.5


def denormalize(board):
    return (board + 0.5) * 9
