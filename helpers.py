import pandas as pd
import numpy as np
from pynput import mouse

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


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

        # Apply any image transformations here if needed
        if self.transform:
            pixels = self.transform(pixels)

        return pixels, label


class ImageDataLoader:
    @staticmethod
    def load_data(path):
        data = pd.read_csv(path, sep=";")
        data = data.sample(frac=1).reset_index(drop=True)
        pixels = data["Pixels"].tolist()
        labels = data["Labels"].tolist()

        image_size = (28, 28)

        numbers = []
        for pixel_sequence in pixels:
            pixel = [int(pixel) for pixel in pixel_sequence.split(" ")]
            pixel = np.asarray(pixel).reshape(*image_size)

            pixel = np.resize(pixel.astype("uint8"), image_size)
            numbers.append(pixel.astype("uint8"))

        numbers = np.asarray(numbers)

        train_numbers = numbers[: int(0.8 * len(numbers))]
        train_labels = labels[: int(0.8 * len(labels))]
        test_numbers = numbers[int(0.8 * len(numbers)) :]
        test_labels = labels[int(0.8 * len(labels)) :]

        return (train_numbers, train_labels), (test_numbers, test_labels)


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
