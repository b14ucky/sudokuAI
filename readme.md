# Sudoku Solver

## Overview

This Sudoku Solver application employs computer vision techniques along with a Convolutional Neural Network (CNN) to recognize and solve Sudoku puzzles. The workflow begins with capturing a screenshot of the Sudoku board, after which the CNN model is utilized to identify the numbers in each cell. The solution is then computed using a deep learning approach.
Initially, I attempted to create a Sudoku-solving model using reinforcement learning, with the goal of achieving a high win rate. However, when the success rate did not meet expectations, I explored alternative solutions. During my research, I came across an article (link [here](https://towardsdatascience.com/solving-sudoku-with-convolution-neural-network-keras-655ba4be3b11)) that recommended the use of Convolutional Neural Networks (CNNs) for Sudoku problem-solving. Inspired by this approach, I revamped the model and transitioned to a CNN-based methodology.
The Sudoku puzzles for this application can be accessed through my Sudoku app, available on [GitHub](https://github.com/b14ucky/sudoku-final-project).

## Usage

To use the script, follow these steps:

1. Clone the repository:
   
```bash
git clone https://github.com/b14ucky/sudokuAI.git
cd sudokuAI
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python main.py
```

4. Position your mouse one pixel below the top-left corner of the board (to ensure 100% accuracy in board recognition) and click the left mouse button. The script will then recognize the board and solve the puzzle. The solution might not be 100% accurate, but it will be close.

## Code Structure

- **board_recognition.py**: Contains the `BoardRecognizer` class, utilizing computer vision and a Convolutional Neural Network (CNN) to recognize individual cells on the Sudoku board.

- **game_player.py**: Defines the `GamePlayer` class that facilitates the interaction with the Sudoku game.

- **helpers.py**: Contains utility functions, including `MouseListener` and `normalize`.

- **models.py**: Defines the `SudokuSolverModel` class for Sudoku sovling and `NumberRecognitionModel`, a PyTorch-based CNN for recognizing the board.

- **main.py**: The main script orchestrating the entire Sudoku solving process.

## Workflow

1. The script waits for a mouse click to capture the position of the Sudoku board.

2. A screenshot is taken, and the CNN model recognizes each cell on the Sudoku board, determining the numbers present.

3. The Sudoku solver model is loaded, and the puzzle is solved using deep learning.

4. The solution is applied to the game interface, and potential mistakes are checked.

5. The process stops after three mistakes or when the puzzle is completely solved.

## Dataset

The model was trained on 1 million sudoku puzzles. You can find the dataset [here](https://www.kaggle.com/datasets/bryanpark/sudoku).

## References

Article: https://towardsdatascience.com/solving-sudoku-with-convolution-neural-network-keras-655ba4be3b11

Dataset: https://www.kaggle.com/datasets/bryanpark/sudoku
