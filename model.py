import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

config = {
    'batch_size': 64
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Iterate through a batch to check the shape of data
for images, labels in train_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class USPSClassification(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(USPSClassification, self).__init__()
        self.layers = nn.ModuleList()
        input_size = input_size

        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, size))
            self.layers.append(nn.ReLU())
            input_size = size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)

        return x

class USPSTrainer(nn.Module):
  def __init__(self, model, train_loader, test_loader):
    super().__init__()
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(model.parameters(), lr=0.001)

  def cnn_train(self):
    num_epochs = 10
    self.model = self.model.to(device)
    for epoch in range(num_epochs):
        self.model.train()
        running_loss = 0.0
        model_predictions = []  # Store model predictions
        ground_truth_labels = []  # Store ground truth labels
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(self.train_loader)}")

  def evaluate(self, iscnn=False):
    self.model.eval()
    self.model = self.model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in self.test_loader:
            images, labels = images.to(device), labels.to(device)

            if not iscnn:
              images = images.view(images.size(0), -1)  # Flatten the images

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total}%")



class USPSCNN(nn.Module):
    def __init__(self):
        super(USPSCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 8*8*32
        x = self.pool(self.relu(self.conv2(x))) # 4*4*64
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

cnn_model = USPSCNN()
model_trainer = USPSTrainer(cnn_model, train_loader, test_loader)
writer = SummaryWriter(f"logs/CNN")

model_trainer.cnn_train()

model_trainer.evaluate(iscnn=True)

writer.close()

def evaluate_model(model, test_loader, iscnn=False):
    all_labels = []
    all_predictions = []
    model = model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

            if not iscnn:
              images = images.view(images.size(0), -1)  # Flatten the images

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, precision, recall, conf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

accuracy_cnn, precision_cnn, recall_cnn, conf_matrix_cnn = evaluate_model(cnn_model, test_loader, iscnn=True)

print("cnn Model Metrics:")
print("Accuracy:", accuracy_cnn)
print("Precision:", precision_cnn)
print("Recall:", recall_cnn)
print("Confusion Matrix:")
print(conf_matrix_cnn)

plt.figure()
plot_confusion_matrix(conf_matrix_cnn, classes=class_names, normalize=False, title='Normalized confusion matrix')
plt.show()


