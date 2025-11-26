import numpy as np
from matplotlib import pyplot as plt

n_epochs = 7
train_loss = np.load('results/plots/train_loss3.npy')[:n_epochs]
val_loss = np.load('results/plots/val_loss3.npy')[:n_epochs]

train_loss = [0 if i<0 else i for i in train_loss]
val_loss = [0 if i<0 else i for i in val_loss]

plt.figure(dpi=200)
plt.plot(train_loss, 'o-r')
plt.plot(val_loss, 'o-b')
plt.legend(['Training loss', 'Validation loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss, L(k)')
plt.savefig('lossplot.png')