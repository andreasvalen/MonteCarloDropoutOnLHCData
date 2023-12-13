import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

def _train_model(model, loader, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs = inputs.unsqueeze(1)
            labels = labels.float().unsqueeze(1)

            model.optimizer.zero_grad()

            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                running_loss = 0.0
        print(f"Epoch {epoch + 1} done")
    print('Finished Training')

def _predictOnData(model, dataLoader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    _preds = np.empty((0, 1))
    _preds_01 = np.empty((0, 1))
    _labels = np.empty((0, 1))

    with torch.no_grad():
        for data in dataLoader: 
            inputs, labels = data
            inputs = inputs.unsqueeze(1)
            labels = labels.float().unsqueeze(1)

            outputs = model(inputs)  # Use self to refer to the model

            # Treat the output as a probability using the sigmoid function
            probabilities = torch.sigmoid(outputs)
            _preds = np.append(_preds, probabilities.numpy())
            _labels = np.append(_labels, labels.numpy())

            # Threshold the probabilities to get predicted labels (0 or 1)
            predicted = (probabilities > 0.5).float()
            _preds_01 = np.append(_preds_01, predicted.numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    return _preds, _preds_01, _labels

def _adjust_dropout(model, new_p):
    # Adjust the dropout rate for all dropout layers
    for layer in model.children():
        if isinstance(layer, nn.Dropout):
            layer.p = new_p


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 channel as input
        self.dropout1 = nn.Dropout(p=0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout3 = nn.Dropout(p=0.1)
        
        # Adjusted for 25x25 input images:
        self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.dropout4 = nn.Dropout(p=0.1)        
        self.fc2 = nn.Linear(120, 84)
        self.dropout5 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(84, 1)

        # Initialize optimizer and loss function
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.dropout4(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def train_model(self, loader, epochs=10):
        _train_model(self, loader, epochs)

    def predictOnData(self, dataLoader):
        return _predictOnData(self, dataLoader)

    def adjust_dropout(self, new_p):
        _adjust_dropout(self, new_p)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BigCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        # More convolutional layers with increased filters
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(40, 80, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(80)
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        # More fully connected layers
        self.fc1 = nn.Linear(80 * 6 * 6, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, 1)
        # Dropout
        self.dropout = nn.Dropout(p=0.1)
        # Optimizer and loss function
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
    def train_model(self, loader, epochs=10):
        _train_model(self, loader, epochs)

    def predictOnData(self, dataLoader):
        return _predictOnData(self, dataLoader)

    def adjust_dropout(self, new_p):
        _adjust_dropout(self, new_p)


