import pyautogui
from time import sleep


class GamePlayer:
    def __init__(self, board_left_corner_coordinates, board):
        self.board_left_corner_coordinates = board_left_corner_coordinates
        self.in_game_mistakes = 0
        self.board = board

    def insert_value(self, column_number, row_number, value):
        # check if the cell is empty and if the value will be valid in that cell
        if self.board[row_number][column_number] != 0:
            reward = -20
            return (
                reward,
                True if self.in_game_mistakes >= 3 or self.board_filled() else False,
                self.in_game_mistakes,
            )

        self.board[row_number][column_number] = value

        if not self.validate_board(column_number, row_number):
            reward = -10
            return (
                reward,
                True if self.in_game_mistakes >= 3 or self.board_filled() else False,
                self.in_game_mistakes,
            )

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
            self.in_game_mistakes += 1
            return (
                reward,
                True if self.in_game_mistakes >= 3 or self.board_filled() else False,
                self.in_game_mistakes,
            )
        reward = 20 if self.board_filled() else 10

        return (
            reward,
            True if self.in_game_mistakes >= 3 or self.board_filled() else False,
            self.in_game_mistakes,
        )

    def validate_board(self, column_number, row_number):
        def is_valid(nums):
            seen = set()
            for num in nums:
                if num != 0 and num in seen:
                    return False
                seen.add(num)
            return True

        for row in self.board:
            if not is_valid(row):
                return False

        for col in range(9):
            column = [self.board[row][col] for row in range(9)]
            if not is_valid(column):
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = [
                    self.board[row][col]
                    for row in range(i, i + 3)
                    for col in range(j, j + 3)
                ]
                if not is_valid(subgrid):
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

    def board_filled(self):
        for row in self.board:
            for value in row:
                if value == 0:
                    return False
        return True

    def restart_game(self):
        x, y = self.board_left_corner_coordinates
        pyautogui.mouseDown(x + 225, y + 210, button="left")
        pyautogui.mouseUp()
