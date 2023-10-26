import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter
from tqdm import tqdm

class Lenet(nn.Module):
  def __init__(self, input_shape, num_classes):
    super(Lenet, self).__init__()
    self.input_shape = input_shape
    self.num_classes = num_classes
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    
    self.flatten = nn.Flatten()
    
    self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=84)
    self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    self.act = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    y = self.act(self.conv1(x))
    y = self.pool1(y)
    y = self.act(self.conv2(y))
    y = self.pool2(y)
    y = self.flatten(y)
    y = self.act(self.fc1(y))
    y = self.act(self.fc2(y))
    y = self.fc3(y)
    y = self.softmax(y)
    return y
  
  def load_from_pth(self, path):
    self.load_state_dict(torch.load(path)) 
    
class Trainer:
  def __init__(self, model, device, num_epochs=100, batch_size=32, lr=0.001):
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.device = device
     
    self.model = model 
    self.model.to(device)
    
    self.loss_fn = nn.NLLLoss() 
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    self.train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    self.test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    self.writer = SummaryWriter()
    self.best_acc = 0.0
    
  def train(self):
    for epoch in range(self.num_epochs):
      self.model.train()
      
      loss = 0.0
      loop = tqdm((self.train_loader), total=len(self.train_loader))
      for i, (x, y) in enumerate(loop):
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        y_pred = torch.log(y_pred)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
  
        loop.set_description('Epoch[{}:{}]: Iteration: {}, Loss: {}'.format(epoch, self.num_epochs, i, loss))
        
      acc = self.test()
      print('Epoch: {}, Accuracy: {}'.format(epoch, acc))
      self.writer.add_scalar('Accuracy', acc, epoch)
      self.writer.add_scalar('Loss', loss, epoch)
      if acc > self.best_acc:
        self.best_acc = acc
        torch.save(self.model.state_dict(), './best_model.pth')
        print("Model saved")
          
   
  def test(self):
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for x, y in self.test_loader:
        total += x.shape[0]
        x = x.to(self.device)
        y_pred = self.model(x)
        y_pred = y_pred.argmax(dim=1).detach().cpu()
        correct += torch.sum(y_pred == y).item()
    
    acc = correct / total    
    return acc

def eval():
  import matplotlib.pyplot as plt
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Lenet(input_shape=(1, 28, 28), num_classes=10)
  model.load_from_pth('./best_model.pth')
  model.to(device)
  model.eval()
  
  test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
  
  # 计算测试集的准确率
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)
  total = 0
  correct = 0
  for x, y in test_loader:
    total += x.shape[0]
    x = x.to(device)
    y_pred = model(x)
    y_pred = y_pred.argmax(dim=1).detach().cpu()
    correct += torch.sum(y_pred == y).item() 
  
  acc = correct/total
  print("Accuracy in test dataset: {}".format(acc))  
  
  # 随机选取20张图片进行测试
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
  for i, (x, y) in enumerate(test_loader):
    if i == 20:
      break
    x = x.to(device)
    y_pred = model(x)
    y_pred = y_pred.argmax(dim=1).detach().cpu().numpy()
    plt.subplot(4, 5, i+1)
    plt.imshow(x[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    color = 'red' if y_pred[0] != y[0] else 'green'
    plt.title('Predict: {}'.format(y_pred[0]), color=color)
    plt.axis('off')
  
  plt.show()    


def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Lenet(input_shape=(1, 28, 28), num_classes=10)
  trainer = Trainer(model, device, num_epochs=10, batch_size=512, lr=0.001)
  trainer.train()    
  
if __name__ == '__main__':
  train()
  