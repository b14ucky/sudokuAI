from models import NumberRecognitionTrainer
from helpers import CustomDataset
from torch.utils.data import DataLoader, random_split

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

trainer = NumberRecognitionTrainer(train_loader, test_loader, lr=0.0001, momentum=0.5)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# trainer.load()
# trainer.train(epochs=5, log_interval=10)
# trainer.save()
# trainer.plot()

trainer.load()
trainer.test_prediction(
    example_data, example_targets, number_of_predictions=len(example_data)
)
