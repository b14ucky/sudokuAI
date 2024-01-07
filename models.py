import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class NumberRecognitionNet(nn.Module):
    def __init__(self):
        super(NumberRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NumberRecognitionModel:
    def __init__(self, lr=0.01, momentum=0.5):
        self.net = NumberRecognitionNet()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.train_counter = []
        self.train_losses = []
        self.test_counter = []
        self.test_losses = []

    def train(self, train_loader, test_loader, epochs=10, log_interval=10):
        self.test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]
        self.accuracy(test_loader)
        for epoch in range(epochs):
            self.net.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch: {epoch + 1} / {epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

                    self.train_losses.append(loss.item())
                    self.train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                    )

            self.accuracy(test_loader)

        print("Finished Training")

    def accuracy(self, test_loader):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.net(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                prediction = output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(target.data.view_as(prediction)).sum()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
        print(
            f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)"
        )

    def test_prediction(
        self, prediction_data, prediction_target, number_of_predictions=5
    ):
        for _ in range(10):
            wrong = 0
            with torch.no_grad():
                for i in range(number_of_predictions):
                    output = self.net(prediction_data[i].view(1, 1, 28, 28))
                    _, predicted = torch.max(output.data, 1)
                    if predicted[0] != prediction_target[i].item():
                        print(
                            f"Predicted: {predicted[0]}, Actual: {prediction_target[i].item()}"
                        )
                        wrong += 1

            print(f"Wrong: {wrong}/{number_of_predictions}")

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def plot(self):
        plt.plot(self.train_counter, self.train_losses, color="blue")
        plt.scatter(self.test_counter, self.test_losses, color="red")
        plt.legend(["Train Loss", "Test Loss"], loc="upper right")
        plt.xlabel("number of training examples seen")
        plt.ylabel("negative log likelihood loss")
        plt.show()
