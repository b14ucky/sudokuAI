import pyautogui
from time import sleep


class GamePlayer:
    def __init__(self, board_left_corner_coordinates):
        self.board_left_corner_coordinates = board_left_corner_coordinates
        self.mistakes = 0

    def insert_value(self, board, column_number, row_number, value):
        # check if the cell is empty and if the value will be valid in that cell
        if board[row_number][column_number] != 0:
            reward = -10
            return reward, False
        if not self.validate_board(board, column_number, row_number, value):
            reward = -10
            return reward, False

        # insert the value into the cell
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(
            x + 25 + column_number * 50, y + 25 + row_number * 50, button="left"
        )
        pyautogui.mouseUp()
        sleep(0.1)
        pyautogui.press(str(value))

        # check if the value was inserted correctly
        if self.check_for_mistake(column_number, row_number):
            reward = -10
            self.mistakes += 1
            return reward, False
        reward = 10 if self.board_filled(board) else 1

        return reward, True

    def validate_board(self, board, column_number, row_number):
        def is_valid(nums):
            seen = set()
            for num in nums:
                if num != 0 and num in seen:
                    return False
                seen.add(num)
            return True

        for row in board:
            if not is_valid(row):
                board[row_number][column_number] = 0
                return False

        for col in range(9):
            column = [board[row][col] for row in range(9)]
            if not is_valid(column):
                board[row_number][column_number] = 0
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = [
                    board[row][col]
                    for row in range(i, i + 3)
                    for col in range(j, j + 3)
                ]
                if not is_valid(subgrid):
                    board[row_number][column_number] = 0
                    return False

        return True

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

    def board_filled(self, board):
        for row in board:
            for value in row:
                if value == 0:
                    return False
        return True

    def restart_game(self):
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(x + 225, y + 210, button="left")
        pyautogui.mouseUp()
