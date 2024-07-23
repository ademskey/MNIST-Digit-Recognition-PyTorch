
import torch
import torch.nn as nn  # Import neural networks (nn)
import torch.nn.functional as F  # Import nn functionality
import torch.optim as optim  # Import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Use torchvision to download training data
train_data = datasets.MNIST(
    root = 'data',
    train = True, # Use ToTensor to define the transformation method
    transform = ToTensor(),
    download = True
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)

# Define pytorch loaders
loaders = {
    'train': DataLoader(train_data, batch_size = 100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size = 100, shuffle=True, num_workers=1)
}

# Define neural network architecture
class CNN(nn.Module):  # define nn as convolutional neural network
    # Define initialization
    def __init__(self):
        super(CNN, self).__init__()

        # Create NN layers
        # Create 2d layer 1 channnel in, 10 out, 5 kernal size
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d() #regularization layer
        self.fc1 = nn.Linear(320, 50) # Fully connected/dense layer
        self.fc2 = nn.Linear(50, 10) # 10 outputs for digit class

    def forward(self, x): # defines activation function
        # Calls relu which is rectify linear unit activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_drop(self.conv2(x)), 2)))
        print(x.shape)
        x.view(-1, 320) #20 * 4 * 4 = 320
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x, dim=1) # dim 

# Configure device to detect if NVIDIA cuda enabled gpu is avaliable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assign device to NN
model = CNN().to(device)

# Configure optimize for model learning (load parameters and learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Define training process
def train(epoch):
    model.train() # Put model in training mode
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Zero out all gradients for each batch before back prop
        output = model(data)

        # Calculate loss, backward propogate, and optimize
        loss = loss_fn(output, target) # Calculate error from desired error
        loss.backward() # Do backward propogation for improvement
        optimizer.step()  # Do optimizer step

        if batch_idx % 25 == 0:# Every 25
            # Fancy print statement
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

            
def test():
    model.eval() # put model into eval mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): # Disable gradient function and no back prop for test
        for data, target in loaders['test']:
            data, target = data.to(device, target.to(device))
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n)')
    
if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()
