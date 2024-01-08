from models import NumberRecognitionModel
from helpers import CustomDataset
from torch.utils.data import DataLoader, random_split

model = NumberRecognitionModel(lr=0.001)

n_epochs = 10
batch_size_train = 64
batch_size_test = 500

custom_dataset = CustomDataset("./dataset/numbers.csv")

train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size

train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size_train, shuffle=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# model.load("model.pth")
# model.train(train_loader, test_loader, epochs=50)
# model.save("model.pth")
# model.plot()

model.load("model.pth")
model.test_prediction(
    example_data, example_targets, number_of_predictions=len(example_data)
)
