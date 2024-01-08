from helpers import CustomDataset
import matplotlib.pyplot as plt
from time import sleep

custom_dataset = CustomDataset("numbers.csv")

for i in range(len(custom_dataset)):
    plt.imshow(custom_dataset[i][0].reshape((28, 28)), cmap="gray")
    plt.title(f"Label: {custom_dataset[i][1]}")
    plt.text(5, 5, i, color="red", fontsize=20)
    plt.show()
