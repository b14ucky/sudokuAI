import os
import pyautogui
import cv2

from mouse_listener import MouseListener

mouse_listener = MouseListener()
mouse_listener.wait_for_click()

num = "seven"

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

    output_dir = num
    os.makedirs(output_dir, exist_ok=True)

    for i, cell in enumerate(cells):
        # Resize the cell to 28x28
        cell_resized = cv2.resize(cell, (28, 28))

        # Save the cell image in the output directory
        cell_filename = os.path.join(output_dir, f"{num}_{0 + i}.png")
        cv2.imwrite(cell_filename, cell_resized)

    print(f"Cell images saved in the '{num}' directory.")
