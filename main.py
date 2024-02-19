from board_recognition import BoardRecognizer
from game_player import GamePlayer
from helpers import MouseListener, normalize
from models import SudokuSolverModel
import pyautogui
import torch


def main():
    listener = MouseListener()
    model = SudokuSolverModel()
    model.load()

    mouse_listener = MouseListener()
    mouse_listener.wait_for_click()

    if mouse_listener.mouse_clicked:
        x, y = pyautogui.position()
        board = torch.tensor(BoardRecognizer().recognize_board(x, y))
        mask = board == 0
        game_player = GamePlayer((x, y))

        board = normalize(board.view(1, 1, 9, 9))
        solution = model.predict(board)

        solution = solution * mask

        mistakes = 0
        for n in range(81):
            i = n % 9
            j = n // 9
            if solution[j][i].item() != 0:
                game_player.insert_value(i, j, solution[j][i].item())
                solution[j][i] = 0
                if game_player.check_for_mistake(i, j):
                    mistakes += 1
                    if mistakes == 3:
                        break


if __name__ == "__main__":
    main()
