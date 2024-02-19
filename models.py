import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from helpers import normalize, denormalize


class NumberRecognitionModel(nn.Module):
    def __init__(self):
        super(NumberRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def load(self, file_name="NRmodel"):
        self.load_state_dict(torch.load(f"{file_name}.pth"))

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            output = self(data.view(1, 1, 28, 28))
            _, predicted = torch.max(output.data, 1)
            return predicted[0].item()


class SudokuSolverModel(nn.Module):
    def __init__(self):
        super(SudokuSolverModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding="same")
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same")
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1), padding="same")

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 81, 81 * 9)
        self.reshape = lambda x: x.view(-1, 9, 81)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.reshape(x)

        return x

    def predict(self, data):
        self.eval()
        sample = data.clone()

        with torch.no_grad():
            while True:
                output = self(sample.view(1, 1, 9, 9))
                output = F.softmax(output, dim=2)
                output = output.squeeze()

                prediction = torch.max(output, dim=0).indices + 1
                prediction = prediction.view(9, 9)
                probability = torch.max(output, dim=0).values.view(9, 9)

                sample = denormalize(sample).view(9, 9)
                mask = sample == 0

                if mask.sum() == 0:
                    break

                new_probability = probability * mask

                index = torch.argmax(new_probability.view(-1))
                row, column = index // 9, index % 9

                value = prediction[row, column]
                sample[row, column] = value
                sample = normalize(sample)

        return torch.round(sample).type(torch.int)

    def load(self, file_name="solver"):
        self.load_state_dict(torch.load(f"{file_name}.pth"))


class NumberRecognitionTrainer:
    def __init__(self, train_loader, test_loader, lr=0.01, momentum=0.5):
        self.model = NumberRecognitionModel()
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # data for plotting
        self.train_counter = []
        self.train_losses = []
        self.test_counter = []
        self.test_losses = []

    def train(self, epochs=10, log_interval=10):
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(epochs + 1)]

        self.accuracy()

        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch: {epoch + 1} / {epochs} [{batch_idx * len(data) if len(data) == 64 else len(self.train_loader.dataset) - len(data)}/{len(self.train_loader.dataset)} ({100.0 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

                    self.train_losses.append(loss.item())
                    self.train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset))
                    )

            self.accuracy()

        print("Finished Training")

    def accuracy(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                prediction = output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(target.data.view_as(prediction)).sum()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print(
            f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({100.0 * correct / len(self.test_loader.dataset):.0f}%)"
        )

    def test_prediction(self, prediction_data, prediction_target, number_of_predictions=5):
        self.model.eval()
        for _ in range(10):
            wrong = 0
            with torch.no_grad():
                for i in range(number_of_predictions):
                    output = self.model(prediction_data[i].view(1, 1, 28, 28))
                    _, predicted = torch.max(output.data, 1)
                    if predicted[0] != prediction_target[i].item():
                        print(f"Predicted: {predicted[0]}, Actual: {prediction_target[i].item()}")
                        wrong += 1

            print(f"Wrong: {wrong}/{number_of_predictions}")

    def save(self, file_name="NRmodel"):
        torch.save(self.model.state_dict(), f"{file_name}.pth")
        torch.save(self.optimizer.state_dict(), f"{file_name}-optimizer.pth")

    def load(self, file_name="NRmodel"):
        self.model.load_state_dict(torch.load(f"{file_name}.pth"))
        self.optimizer.load_state_dict(torch.load(f"{file_name}-optimizer.pth"))

    def plot(self):
        plt.plot(self.train_counter, self.train_losses, color="blue")
        plt.scatter(self.test_counter, self.test_losses, color="red")
        plt.legend(["Train Loss", "Test Loss"], loc="upper right")
        plt.xlabel("number of training examples seen")
        plt.ylabel("negative log likelihood loss")
        plt.show()


class SudokuSolverTrainer:
    def __init__(self, train_loader, test_loader, lr=0.001):
        self.model = SudokuSolverModel()
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=2, log_interval=100):
        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data.unsqueeze(1))
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch: {epoch + 1} / {epochs} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100.0 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

        print("Finished Training")

    def save(self, file_name="solver"):
        torch.save(self.model.state_dict(), f"{file_name}.pth")
        torch.save(self.optimizer.state_dict(), f"{file_name}-optimizer.pth")

    def load(self, file_name="solver"):
        self.model.load_state_dict(torch.load(f"{file_name}.pth"))
        self.optimizer.load_state_dict(torch.load(f"{file_name}-optimizer.pth"))
