import numpy as np
import copy
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, datasets
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
import matplotlib.pyplot as plt


# Training Options
#==============================================================================================================
EPOCH = 50             # Epoch
batch = 100            # mini-batch size
n = 50                 # the number of nodes in bottleneck layer (Z)
lr = 0.005             # learning rate
view = 15              # the number of sample images
gamma = 2              # constant for MI
alpha = 1.01           # Renyi's alpha-entropy
time_int = 'Iteration' # Time interval of Information Plane : Iteration
# time_int = 'Epoch'     # Time interval of Information Plane : Epoch
epsilon = 10**(-8)     # divergence regulator
DEVICE = "cpu"
#==============================================================================================================

# Data Load
#==============================================================================================================
trainset = datasets.EMNIST(root = './.data/', split='byclass', train = True, download = True, transform = transforms.ToTensor())
testset = datasets.EMNIST(root = './.data/', split='byclass', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=batch, shuffle = True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size=batch, shuffle = True, num_workers=0)

lower_index = np.where(trainset.targets > 35)
lower_data = trainset.data[lower_index]
lower_targets = trainset.targets[lower_index]
trainset.data = lower_data[:160000]
trainset.targets = (lower_targets - 36)[:160000]

lower_index_test = np.where(testset.targets > 35)
lower_data_test = testset.data[lower_index_test]
lower_targets_test = testset.targets[lower_index_test]
testset.data = lower_data_test
testset.targets = (lower_targets_test - 36)
#==============================================================================================================


class Tied_AE(nn.Module):
    
    def __init__(self, n):
        super(Tied_AE, self).__init__()
        
        self.fc1 = nn.Linear(28*28,n)
        self.fc2 = nn.Linear(n,28*28)
        self.fc2.weight.data = self.fc1.weight.data.transpose(0,1)
        
    def forward(self,x):
        
        X = x.view(-1,28*28)
        Z = torch.sigmoid(self.fc1(X))
        Y = torch.sigmoid(self.fc2(Z))

        return Y

tae = Tied_AE(n).to(DEVICE)
optimizer = torch.optim.Adam(tae.parameters(), lr = lr)
MSE = nn.MSELoss()  

def train(tae, train_loader, history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss):
    
    tae.train()
    for step, (x,label) in enumerate(train_loader):
        x = x.view(-1,28*28).to(DEVICE)
        y = x.view(-1,28*28).to(DEVICE)
        label = label.to(DEVICE)

        Y = tae(x)
            
        W1 = tae.fc1.weight.data.detach().numpy()
        b1 = tae.fc1.bias.data.detach().numpy()
        W2 = tae.fc2.weight.data.detach().numpy()
        b2 = tae.fc2.bias.data.detach().numpy()

        history_W1.append(copy.deepcopy(W1))
        history_b1.append(copy.deepcopy(b1))
        history_W2.append(copy.deepcopy(W2))
        history_b2.append(copy.deepcopy(b2))

        trainloss = MSE(Y, y)
        history_trainloss.append(trainloss.detach().numpy())
        
        test_data = testset.data.view(-1,784).type(torch.FloatTensor)/255.
        output = tae(test_data)
        testloss = MSE(output, test_data)    
        history_testloss.append(testloss.detach().numpy())      
        
        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()
        
    return (history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss)

def sigmoid(x):
    return expit(x)

def Ent(X,gamma,alpha):
    N = np.size(X,0)
    d = np.size(X,1)
    sigma = gamma*N**(-1/(4+d))
    X_norm = X
    pairwise_dist = squareform(pdist(X_norm, 'euclidean'))
    K = np.exp(-pairwise_dist**2/(2*sigma**2))    
    A = 1/N*K
    _, eigenv, _ = np.linalg.svd(A)
    S = 1/(1-alpha)*np.log2(np.sum(eigenv**alpha)+epsilon).real
    return A, S

def MI(X,Y,gamma,alpha):
    A_X, S_X = Ent(X,gamma,alpha)
    A_Y, S_Y = Ent(Y,gamma,alpha)
    A_XY = A_X*A_Y/np.trace(A_X*A_Y)
    _, eigenv, _ = np.linalg.svd(A_XY)
    S_XY = 1/(1-alpha)*np.log2(np.sum(eigenv**alpha)+epsilon).real  
    S = S_X + S_Y - S_XY
    return S, S_XY


def encoder(test, W1, b1):
    Z = sigmoid(np.einsum('ij,jk->ik', test, W1.T) + b1)
    return Z

def decoder(Z, W2, b2):
    output = sigmoid(np.einsum('ij,jk->ik', Z, W2.T) + b2)
    return output

def IP(history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss):
    
    if time_int == 'Epoch':
        step = EPOCH
        ind = np.linspace(0,len(history_trainloss)*(1-1/step),step)    
    elif time_int == 'Iteration':
        jump = 1
        step = np.int(len(history_trainloss)/jump)
        ind = np.linspace(0,len(history_trainloss)-jump,step)

    I_XZ_cont = np.zeros((step,))
    I_ZY_cont = np.zeros((step,))

    train_E_cont = np.zeros((step,))
    test_E_cont = np.zeros((step,))

    EMNIST_test = testset.data.view(-1,28*28).type(torch.FloatTensor)/255.
    EMNIST_test = EMNIST_test.detach().numpy()
    
    for j in range(step):
        i = np.int(ind[j])
        W1 = history_W1[i]
        b1 = history_b1[i]
        b1 = np.reshape(b1, (1,len(b1)))
        W2 = history_W2[i]
        b2 = history_b2[i]
        b2 = np.reshape(b2, (1,len(b2)))

        train_E = history_trainloss[i]
        test_E = history_testloss[i]
        
        X = EMNIST_test[:batch,:]
        Z = encoder(X, W1, b1)
        Y = decoder(Z, W2, b2)
       
        I_XZ, H_XZ = MI(X,Z,gamma,alpha)
        I_ZY, H_ZY = MI(Z,Y,gamma,alpha)
       
        I_XZ_cont[j] = I_XZ
        I_ZY_cont[j] = I_ZY
         
        train_E_cont[j] = train_E
        test_E_cont[j] = test_E


    # Information plane & Train/Test Loss
    D = 7
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(1,2,1)
    plt.plot(np.log10(train_E_cont), label='Train')
    plt.plot(np.log10(test_E_cont), label='Test')
    plt.title('Train/Test loss', fontsize=15)
    plt.legend()
    plt.xlabel('{}'.format(time_int))
    plt.ylabel('log(Loss)')
    xx = np.linspace(0,D,500)
    yy = np.linspace(0,D,500)
    num = np.linspace(0,step,step)
    ax2 = plt.subplot(1,2,2)
    plt.plot(xx, yy, 'k--')
    im = plt.scatter(I_XZ_cont, I_ZY_cont, c=num, cmap='rainbow')
    plt.title('Information Plane of Tied Autoencoder', fontsize=15)
    plt.xlabel(r'$I(X;Z)$', fontsize=13)
    plt.ylabel(r"$I(Z;X')$", fontsize=13)
    b_ax = fig.add_axes([0.92, 0.13, 0.015, 0.75])
    bar = fig.colorbar(im, cax=b_ax)
    bar.set_label('{}'.format(time_int))
    plt.show()


def main():
    history_W1 = []
    history_b1 = []
    history_W2 = []
    history_b2 = []

    history_trainloss = []
    history_testloss = []
    sample_data = trainset.data[:view].view(-1,28*28)
    sample_data = sample_data.type(torch.FloatTensor)/255.
        
    print('Training Starts!')
    for epoch in range(1,EPOCH+1):
        history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss = \
        train(tae, train_loader, history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss)
            
        sample_x = sample_data.to(DEVICE)
        sample_y = tae(sample_x)

        if epoch == EPOCH:
            f,a = plt.subplots(2,view,figsize=(view,2))
            for i in range(view):
                img = np.reshape(sample_x.data.numpy()[i], (28,28))
                a[0][i].imshow(img.T, cmap='gray')
                a[0][i].set_xticks(()); a[0][i].set_yticks(())

            for i in range(view):
                img = np.reshape(sample_y.data.numpy()[i],(28,28))
                a[1][i].imshow(img.T, cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show()
    print('Training Ends!')

    print('Estimating Mutual Information...')

    IP(history_W1, history_b1, history_W2, history_b2, history_trainloss, history_testloss)
























