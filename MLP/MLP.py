import numpy as np
from tensorboardX import SummaryWriter
from sklearn.datasets import load_breast_cancer

class MLP:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.ouput_size = output_size
    
    self.W1 = np.random.randn(input_size, hidden_size)
    self.b1 = np.zeros((1, hidden_size))
    self.W2 = np.random.randn(hidden_size, output_size)
    self.b2 = np.zeros((1, output_size))
    
    self.writer = SummaryWriter() 
    
    self.z1 = None
    self.a1 = None
    self.z2 = None
    self.a2 = None
   
  def __call__(self, x):
    '''
    forward but not update z1, a1, z2, a2
    '''
    y = np.dot(x, self.W1) + self.b1
    y = self._sigmoid(y)
    y = np.dot(y, self.W2) + self.b2
    y = self._softmax(y)
    return y
    
   
  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
    
  def _softmax(self, x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
  
  def _cross_enropy_loss(self, y_pred, y_target):
    num_samples = y_pred.shape[0]
    log_probs = -np.log(y_pred[np.arange(num_samples), y_target.argmax(axis=1)])
    return np.sum(log_probs) / num_samples    
  
  def forward(self, x):
    '''
    forward
    '''
    self.z1 = np.dot(x, self.W1) + self.b1
    self.a1 = self._sigmoid(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self._softmax(self.z2) 
    return self.a2.copy()
  
  def _backward(self, x, y_target, lr):
    num_samples = x.shape[0]
    
    delta2 = self.a2.copy()
    delta2[range(num_samples), y_target.argmax(axis=1)] -= 1
    delta2 /= num_samples
    
    dW2 = np.dot(self.a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    
    delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1)
    
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    
    self.W2 -= lr * dW2
    self.b2 -= lr * db2
    self.W1 -= lr * dW1
    self.b1 -= lr * db1 
    
  
  def train(self, x, y_target, lr=0.01, epochs=100):
    for epoch in range(epochs):
      # forward
      y_pred = self.forward(x)
      # backward
      loss = self._cross_enropy_loss(y_pred, y_target)
      self._backward(x, y_target, lr)
      print('Epoch: {}, Loss: {}'.format(epoch, loss))
      self.writer.add_scalar('Loss', loss, epoch)
      if epoch % 10 == 0:
        acc = self.test(x, y_target)
        print('Accuracy: {}'.format(acc))
        self.writer.add_scalar('Accuracy', acc, epoch)
      
  def test(self, x, y_target):
    y_pred = self.forward(x)
    accuracy = np.sum(y_pred.argmax(axis=1) == y_target.argmax(axis=1)) / x.shape[0]
    return accuracy
      
      
def main():
  # load data
  data = load_breast_cancer()
  x = data.data
  y = data.target
  
  # split data 
  num_samples = x.shape[0]
  num_train = int(num_samples * 0.8)
  num_test = num_samples - num_train
  
  x_train = x[:num_train]
  y_train = y[:num_train]
  x_test = x[num_train:]
  y_test = y[num_train:]
  
  # one-hot encode
  y_train = np.eye(2)[y_train]
  y_test = np.eye(2)[y_test]
  print(y_train.shape)
  
  model = MLP(input_size=30, hidden_size=100, output_size=2)
  model.train(x_train, y_train, lr=0.01, epochs=300)
  model.test(x_test, y_test)
  
  
if __name__ == '__main__':
  main()