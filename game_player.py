import pyautogui
from time import sleep


class GamePlayer:
    def __init__(self, board_left_corner_coordinates):
        self.board_left_corner_coordinates = board_left_corner_coordinates
        self.in_game_mistakes = 0

    def insert_value(self, column_number, row_number, value):
        # insert the value into the cell
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(x + 25 + column_number * 50, y + 25 + row_number * 50, button="left")
        pyautogui.mouseUp()
        sleep(0.1)
        pyautogui.press(str(value))
        sleep(0.25)

    def check_for_mistake(self, column_number, row_number):
        x, y = self.board_left_corner_coordinates
        color = pyautogui.pixel(x + 25 + column_number * 50, y + 25 + row_number * 50)
        # if the color of the cell is in the shade of red, then the value was inserted incorrectly
        if (
            color[0] >= 134
            and color[0] <= 233
            and color[1] >= 26
            and color[1] <= 48
            and color[2] >= 26
            and color[2] <= 51
        ):
            return True
        return False

    def restart_game(self):
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(x + 225, y + 210, button="left")
        pyautogui.mouseUp()
