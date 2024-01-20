import pyautogui
import cv2
from helpers import MouseListener, display_board
from models import NumberRecognitionModel
from torch import from_numpy
import numpy as np


class BoardRecognizer:
    def __init__(self):
        self.model = NumberRecognitionModel()
        self.model.load()
        self.mouse_listener = MouseListener()

    def recognize_board(self):
        self.mouse_listener.wait_for_click()

        if self.mouse_listener.mouse_clicked:
            x, y = pyautogui.position()
            print(x, y)

            region_to_capture = (x, y, 450, 450)

            screenshot = pyautogui.screenshot(region=region_to_capture)
            screenshot.save("sudoku_board.png")

            image = cv2.imread("sudoku_board.png", cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(
                image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            image = cv2.bitwise_not(image)

            cell_size = image.shape[0] // 9
            cells = [
                image[i : i + cell_size - 5, j : j + cell_size - 5]
                for i in range(0, image.shape[0], cell_size)
                for j in range(0, image.shape[1], cell_size)
            ]

            board = np.array([0] * 81).reshape((9, 9))
            for i, cell in enumerate(cells):
                cell = cv2.resize(cell, (28, 28)).astype(np.float32)
                prediction = self.model.predict(from_numpy(cell))
                board[i // 9][i % 9] = prediction

            display_board(board)

            return board
