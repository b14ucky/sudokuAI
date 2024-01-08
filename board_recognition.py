import pyautogui
import cv2
from helpers import MouseListener
from models import NumberRecognitionModel
from torch import from_numpy
import numpy as np
import matplotlib.pyplot as plt

model = NumberRecognitionModel()

model.load("model.pth")

mouse_listener = MouseListener()
mouse_listener.wait_for_click()

if mouse_listener.mouse_clicked:
    x, y = pyautogui.position()
    print(x, y)

    region_to_capture = (x, y, 450, 450)

    screenshot = pyautogui.screenshot(region=region_to_capture)
    screenshot.save("sudoku_board.png")

    image = cv2.imread("sudoku_board.png", cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
        prediction = model.predict(from_numpy(cell))
        board[i // 9][i % 9] = prediction

    fig, ax = plt.subplots()
    ax.axis("off")

    table = ax.table(
        cellText=board, loc="center", cellLoc="center", colWidths=[0.1] * len(board[0])
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for i in range(len(board[0])):
        for j in range(len(board[0])):
            table[(i, j)].set_height(0.1)

    plt.show()
