import torch
import matplotlib.pyplot as plt
import numpy as np

path = 'snapshots/resnet34_siamese.pth.tar'
checkpoint = torch.load(path)

train_loss_list = checkpoint['train_loss_list']
val_acc_list = checkpoint['val_acc_list']

x = np.asarray(range(len(train_loss_list)))+1
x = x/2070.0
plt.plot(x[3:],train_loss_list[3:])
plt.xlabel('Epochs')
plt.ylabel('Loss(-logp)')
plt.title('Training loss curve')
plt.show()

x = np.asarray(range(40))+1
plt.plot(x, val_acc_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy curve')
plt.show()
