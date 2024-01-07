import pandas as pd
import numpy as np
from pynput import mouse
import torchvision.transforms as transforms
from torch.utils.data import Dataset


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
