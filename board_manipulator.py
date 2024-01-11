import pyautogui
from time import sleep


class BoardManipulator:
    def __init__(self, board_left_corner_coordinates):
        self.board_left_corner_coordinates = board_left_corner_coordinates

    def insert_value(self, column_number, row_number, value):
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(
            x + 25 + column_number * 50, y + 25 + row_number * 50, button="left"
        )
        pyautogui.mouseUp()
        sleep(0.1)
        pyautogui.press(str(value))
