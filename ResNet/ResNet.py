import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

class BottleNeck(nn.Module):
  def __init__(self, input_channels, out_channels, downsample=False, stride=1):
    super(BottleNeck, self).__init__()
    mid_channels = int(out_channels/4)
    self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
    self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(mid_channels)
    self.bn2 = nn.BatchNorm2d(mid_channels)
    self.bn3 = nn.BatchNorm2d(out_channels)
    
    if downsample:
      self.downsample = nn.Sequential(nn.Conv2d(input_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels))
    else:
      self.downsample = False
    
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, X):
    Y = self.relu(self.bn1(self.conv1(X)))
    Y = self.relu(self.bn2(self.conv2(Y)))
    Y = self.bn3(self.conv3(Y))
    if self.downsample:
      X = self.downsample(X)
    Y += X
    return self.relu(Y)

class Backbone(nn.Module):
  def __init__(self, input_channels, layers):
    super(Backbone, self).__init__()
    self.input_shape = input_channels 
    self.layers = layers
    self.stage1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    self.stage2 = self.make_stage(64, 256, self.layers[0], stride=1) 
    self.stage3 = self.make_stage(256, 512, self.layers[1], stride=2)
    self.stage4 = self.make_stage(512, 1024, self.layers[2], stride=2)
    self.stage5 = self.make_stage(1024, 2048, self.layers[3], stride=2)
    
  
  def make_stage(self, input_channels, output_channels, blocks, stride=1):
    layers = [] 
    first_block = True
    for i in range(blocks):
      if first_block:
        first_block = False
        layers.append(BottleNeck(input_channels, output_channels, downsample=True, stride=stride)) 
      else:
        layers.append(BottleNeck(output_channels, output_channels)) 
    
    return nn.Sequential(*layers)    
  
  def forward(self, X):
    Y = self.stage1(X)
    Y = self.stage2(Y)
    Y = self.stage3(Y)
    Y = self.stage4(Y)
    Y = self.stage5(Y)
    return Y

class Head(nn.Module):
  def __init__(self, input_channels, num_classes):
    super(Head, self).__init__()
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(input_channels, num_classes)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, X):
    Y = self.avgpool(X)
    Y = Y.view(Y.size(0), -1)
    Y = self.fc(Y)
    Y = self.softmax(Y) 
    return Y  


class ResNet50(nn.Module):
  def __init__(self, input_shape, num_classes):
    super(ResNet50, self).__init__()
    self.input_shape = input_shape
    self.backbone = Backbone(input_channels=3, layers=[3,4,6,3])
    self.head = Head(input_channels=2048, num_classes=num_classes)
    
  def forward(self, X):
    Y = self.backbone(X)
    Y = self.head(Y)
    return Y
 
class Trainer:
  def __init__(self, model, device, num_epochs=100, batch_size=32, lr=0.001, dataset = 'CIFAR100'):
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.device = device
     
    self.model = model 
    self.model.to(device)
    
    self.loss_fn = nn.CrossEntropyLoss(reduction='mean') 
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    if dataset == 'CIFAR10':
      self.train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
      self.test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    elif dataset == 'FashionMNIST':
      self.train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
      self.test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    self.writer = SummaryWriter()
    self.best_acc = 0.0
    
  def train(self):
    for epoch in range(self.num_epochs):
      self.model.train()
      for i, (x, y) in enumerate(self.train_loader):
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if i % 100 == 0:
          print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss))
        self.writer.add_scalar('Loss', loss, epoch)
        
      acc = self.test()
      print('Epoch: {}, Accuracy: {}'.format(epoch, acc))
      self.writer.add_scalar('Accuracy', acc, epoch)
      if acc > self.best_acc:
        self.best_acc = acc
        torch.save(self.model.state_dict(), './best_model.pth')
          
   
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

def predict():
  import matplotlib.pyplot as plt
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = ResNet50(input_shape=(3,224,224), num_classes=100)
  model.load_from_pth('./best_model.pth')
  model.to(device)
  model.eval()
  
  test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
  
  # predict 20 images and plot them
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

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = ResNet50(input_shape=(3,224,224), num_classes=100)
  trainer = Trainer(model, device, num_epochs=200)
  trainer.train()
  predict()
  
if __name__ == '__main__':
  main()