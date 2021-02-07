import numpy as np
import copy
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
import matplotlib.pyplot as plt


# Training Options
#==============================================================================================================
EPOCH = 50             # Epoch
batch = 100            # mini-batch size
n1 = 256               # the number of nodes in the first hidden layer (E1)
n2 = 128               # the number of nodes in the second hidden layer (E2)
n3 = 50                # the number of nodes in bottleneck layer (Z)
lr = 0.005             # learning rate
view = 15              # the number of sample images
gamma = 2              # constant in kernel bandwidth
alpha = 1.01           # Renyi's alpha-entropy
time_int = 'Iteration' # Time interval of Information Plane : iteration 
# time_int = 'Epoch'     # Time interval of Inforamtion Plane : Epoch
epsilon = 10**(-8)     # divergence regulator
DEVICE = "cpu"
#==============================================================================================================

# Data Load
#==============================================================================================================
trainset = datasets.FashionMNIST(root = './.data/', train = True, download = True, transform = transforms.ToTensor())
testset = datasets.FashionMNIST(root = './.data/', train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=batch, shuffle = True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size=batch, shuffle = True, num_workers=0)
#==============================================================================================================


class VAE(nn.Module):
    
    def __init__(self, n1, n2, n3):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(28*28,n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc31 = nn.Linear(n2,n3)
        self.fc32 = nn.Linear(n2,n3)
        self.fc4 = nn.Linear(n3,n2)
        self.fc5 = nn.Linear(n2,n1)
        self.fc6 = nn.Linear(n1,28*28)
        
    def encode(self, x):
        X = x.view(-1,784)
        E1 = torch.sigmoid(self.fc1(X))
        E2 = torch.sigmoid(self.fc2(E1))
        mu = self.fc31(E2)
        logvar = self.fc32(E2)
        return E1, E2, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std, eps

    def decode(self, z):
        D1 = torch.sigmoid(self.fc4(z))
        D2 = torch.sigmoid(self.fc5(D1))
        Y = torch.sigmoid(self.fc6(D2))
        return D1, D2, Y

    def forward(self, x):
        E1, E2, mu, logvar = self.encode(x)
        Z, eps = self.reparameterize(mu, logvar)
        D1, D2, Y = self.decode(Z)
        return mu, logvar, eps, Z, Y

vae = VAE(n1,n2,n3).to(DEVICE)
optimizer = torch.optim.Adam(vae.parameters(), lr = lr)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = (-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(vae, train_loader, history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, \
          history_W4, history_b4, history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss):
    
    vae.train()
    for step, (x,label) in enumerate(train_loader):
        x = x.view(-1,28*28).to(DEVICE)
        y = x.view(-1,28*28).to(DEVICE)
        label = label.to(DEVICE)

        mu, logvar, eps, Z, Y = vae(x)
            
        W1 = vae.fc1.weight.data.detach().numpy()
        b1 = vae.fc1.bias.data.detach().numpy()
        W2 = vae.fc2.weight.data.detach().numpy()
        b2 = vae.fc2.bias.data.detach().numpy()
        W31 = vae.fc31.weight.data.detach().numpy()
        b31 = vae.fc31.bias.data.detach().numpy()
        W32 = vae.fc32.weight.data.detach().numpy()
        b32 = vae.fc32.bias.data.detach().numpy()
        W4 = vae.fc4.weight.data.detach().numpy()
        b4 = vae.fc4.bias.data.detach().numpy()
        W5 = vae.fc5.weight.data.detach().numpy()
        b5 = vae.fc5.bias.data.detach().numpy()
        W6 = vae.fc6.weight.data.detach().numpy()
        b6 = vae.fc6.bias.data.detach().numpy()
        eps = eps.detach().numpy()

        history_W1.append(copy.deepcopy(W1))
        history_b1.append(copy.deepcopy(b1))
        history_W2.append(copy.deepcopy(W2))
        history_b2.append(copy.deepcopy(b2))
        history_W31.append(copy.deepcopy(W31))
        history_b31.append(copy.deepcopy(b31))
        history_W32.append(copy.deepcopy(W32))
        history_b32.append(copy.deepcopy(b32))
        history_W4.append(copy.deepcopy(W4))
        history_b4.append(copy.deepcopy(b4))
        history_W5.append(copy.deepcopy(W5))
        history_b5.append(copy.deepcopy(b5))
        history_W6.append(copy.deepcopy(W6))
        history_b6.append(copy.deepcopy(b6))
        history_eps.append(copy.deepcopy(eps))

        trainloss = loss_function(Y,x, mu, logvar)
        # trainloss = MSE(Y, y)
        history_trainloss.append(trainloss.detach().numpy())
        
        test_data = testset.data.view(-1,784).type(torch.FloatTensor)/255.
        mu_test, logvar_test, eps_test, Z_test, Y_test = vae(test_data)
        testloss = loss_function(Y_test, test_data, mu_test, logvar_test) 
        history_testloss.append(testloss.detach().numpy())      
        
        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()
        
    return (history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, \
            history_W4, history_b4, history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss)

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

def encoder(test, W1, b1, W2, b2, W31, b31, W32, b32, eps):
    E1 = sigmoid(np.einsum('ij,jk->ik', test, W1.T) + b1)
    E2 = sigmoid(np.einsum('ij,jk->ik', E1, W2.T) + b2)
    mu = np.einsum('ij,jk->ik', E2, W31.T) + b31
    logvar = np.einsum('ij,jk->ik', E2, W32.T) + b32
    std = np.exp(0.5*logvar)
    Z = mu + eps*std
    return E1, E2, mu, std, Z

def decoder(Z, W4, b4, W5, b5, W6, b6):
    D1 = sigmoid(np.einsum('ij,jk->ik', Z, W4.T) + b4)
    D2 = sigmoid(np.einsum('ij,jk->ik', D1, W5.T) + b5)
    output = sigmoid(np.einsum('ij,jk->ik', D2, W6.T) + b6)    
    return D1, D2, output

def IP(history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, history_W4, history_b4,
       history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss):
    
    if time_int == 'Epoch':
        step = EPOCH
        ind = np.linspace(0,len(history_trainloss)*(1-1/step),step)    
    elif time_int == 'Iteration':
        jump = 1
        step = np.int(len(history_trainloss)/jump)
        ind = np.linspace(0,len(history_trainloss)-jump,step)
        
    I_XE1_cont = np.zeros((step,))
    I_E1Y_cont = np.zeros((step,))
    I_XE2_cont = np.zeros((step,))
    I_E2Y_cont = np.zeros((step,))
    I_XZ_cont = np.zeros((step,))
    I_ZY_cont = np.zeros((step,))
    I_XD1_cont = np.zeros((step,))
    I_D1Y_cont = np.zeros((step,))
    I_XD2_cont = np.zeros((step,))
    I_D2Y_cont = np.zeros((step,))
    train_E_cont = np.zeros((step,))
    test_E_cont = np.zeros((step,))

    FMNIST_test = testset.data.view(-1,28*28).type(torch.FloatTensor)/255.
    FMNIST_test = FMNIST_test.detach().numpy()
    
    for j in range(step):
        i = np.int(ind[j])
        W1 = history_W1[i]
        b1 = history_b1[i]
        b1 = np.reshape(b1, (1,len(b1)))
        W2 = history_W2[i]
        b2 = history_b2[i]
        b2 = np.reshape(b2, (1,len(b2)))
        W31 = history_W31[i]
        b31 = history_b31[i]
        b31 = np.reshape(b31, (1,len(b31)))
        W32 = history_W32[i]
        b32 = history_b32[i]
        b32 = np.reshape(b32, (1,len(b32)))
        W4 = history_W4[i]
        b4 = history_b4[i]
        b4 = np.reshape(b4, (1,len(b4)))
        W5 = history_W5[i]
        b5 = history_b5[i]
        b5 = np.reshape(b5, (1,len(b5)))
        W6 = history_W6[i]
        b6 = history_b6[i]
        b6 = np.reshape(b6, (1,len(b6)))
        eps = history_eps[i]
        train_E = history_trainloss[i]
        test_E = history_testloss[i]
        
        X = FMNIST_test[:batch,:]
        E1, E2, mu, std, Z = encoder(X, W1, b1, W2, b2, W31, b31, W32, b32, eps)
        D1, D2, Y = decoder(Z, W4, b4, W5, b5, W6, b6)

        I_XE1, H_XE1 = MI(X,E1,gamma,alpha)
        I_E1Y, H_E1Y = MI(E1,Y,gamma,alpha)
        
        I_XE2, H_XE2 = MI(X,E2,gamma,alpha)
        I_E2Y, H_E2Y = MI(E2,Y,gamma,alpha)
        
        I_XZ, H_XZ = MI(X,Z,gamma,alpha)
        I_ZY, H_ZY = MI(Z,Y,gamma,alpha)
        
        I_XD1, H_XD1 = MI(X,D1,gamma,alpha)
        I_D1Y, H_D1Y = MI(D1,Y,gamma,alpha)
        
        I_XD2, H_XD2 = MI(X,D2,gamma,alpha)
        I_D2Y, H_D2Y = MI(D2,Y,gamma,alpha)

        I_XE1_cont[j] = I_XE1
        I_E1Y_cont[j] = I_E1Y
        
        I_XE2_cont[j] = I_XE2
        I_E2Y_cont[j] = I_E2Y
        
        I_XZ_cont[j] = I_XZ
        I_ZY_cont[j] = I_ZY
        
        I_XD1_cont[j] = I_XD1
        I_D1Y_cont[j] = I_D1Y
        
        I_XD2_cont[j] = I_XD2
        I_D2Y_cont[j] = I_D2Y
          
        train_E_cont[j] = train_E
        test_E_cont[j] = test_E


    # Information plane 
    D = 7
    size = 7
    xx = np.linspace(0,D,500)
    yy = np.linspace(0,D,500)
    num = np.linspace(0,step,step)
    fig = plt.figure(figsize=(12,8))
    suptitle = fig.suptitle('Information Plane of Variational Autoencoder', y=1.01, fontsize='20')
    ax1 = plt.subplot(2,3,1)
    plt.plot(xx, yy, 'k--')
    im = plt.scatter(I_XE1_cont, I_E1Y_cont, c=num, cmap='rainbow', label = 'E1', s=size)
    plt.ylabel(r"$I(T;X')$", fontsize=13)
    ax1.axes.get_xaxis().set_ticks([])
    plt.legend(fontsize='15')
    ax2 = plt.subplot(2,3,2)
    plt.plot(xx, yy, 'k--')
    plt.scatter(I_XE2_cont, I_E2Y_cont, c=num, cmap='rainbow', label = 'E2', s=size)
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    plt.legend(fontsize='15')
    ax3 = plt.subplot(2,3,3)
    plt.plot(xx, yy, 'k--')
    plt.scatter(I_XZ_cont, I_ZY_cont, c=num, cmap='rainbow', label = 'Z', s=size)
    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    plt.legend(fontsize='15')
    ax4 = plt.subplot(2,3,4)
    plt.plot(xx, yy, 'k--')
    plt.scatter(I_XZ_cont, I_ZY_cont, c=num, cmap='rainbow', label = 'Z', s=size)
    plt.xlabel(r'$I(X;T)$', fontsize=13)
    plt.ylabel(r"$I(T;X')$", fontsize=13)
    plt.legend(fontsize='15')
    ax5 = plt.subplot(2,3,5)
    plt.plot(xx, yy, 'k--')
    plt.scatter(I_XD1_cont, I_D1Y_cont, c=num, cmap='rainbow', label = 'D1', s=size)
    plt.xlabel(r'$I(X;T)$', fontsize=13)
    ax5.axes.get_yaxis().set_ticks([])
    plt.legend(fontsize='15')
    ax6 = plt.subplot(2,3,6)
    plt.plot(xx, yy, 'k--')
    plt.scatter(I_XD2_cont, I_D2Y_cont, c=num, cmap='rainbow', label = 'D2', s=size)
    plt.xlabel(r'$I(X;T)$', fontsize=13)
    ax6.axes.get_yaxis().set_ticks([])
    plt.legend(fontsize='15')
    plt.tight_layout()
    b_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    bar = fig.colorbar(im, cax=b_ax)
    bar.set_label('{}'.format(time_int))
    plt.show()

    # DPI & Train/Test Loss
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1)
    plt.plot(I_XE1_cont, label = r'$I(X;E_1)$')
    plt.plot(I_XE2_cont, label = r'$I(X;E_2)$')
    plt.plot(I_XZ_cont, label = 'I(X;Z)')
    plt.xlabel('{}'.format(time_int))
    plt.title('DPI of Encoder', fontsize=15)
    plt.legend()
    ax2 = plt.subplot(1,3,2)
    plt.plot(I_D2Y_cont, label = r'$I(D_2;Y)$')
    plt.plot(I_D1Y_cont, label = r'$I(D_1;Y)$')
    plt.plot(I_ZY_cont, label = 'I(Z;Y)')
    plt.xlabel('{}'.format(time_int))
    plt.title('DPI of Decoder', fontsize=15)
    plt.legend()
    ax3 = plt.subplot(1,3,3)
    plt.plot(np.log10(train_E_cont/batch), label='Train')
    plt.plot(np.log10(test_E_cont/len(testset.data)), label='Test')
    plt.ylabel('log(Loss)')
    plt.xlabel('{}'.format(time_int))
    plt.title('Train/Test Loss', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    history_W1 = []
    history_b1 = []
    history_W2 = []
    history_b2 = []
    history_W31 = []
    history_b31 = []
    history_W32 = []
    history_b32 = []
    history_W4 = []
    history_b4 = []
    history_W5 = []
    history_b5 = []
    history_W6 = []
    history_b6 = []
    history_eps = []
    history_trainloss = []
    history_testloss = []
    sample_data = trainset.data[:view].view(-1,28*28)
    sample_data = sample_data.type(torch.FloatTensor)/255.
        
    print('Training Starts!')
    for epoch in range(1,EPOCH+1):
        history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, \
        history_W4, history_b4, history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss = \
        train(vae, train_loader, history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, \
              history_W4, history_b4, history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss)
            
        sample_x = sample_data.to(DEVICE)
        _,_,_,_, sample_y = vae(sample_x)

        if epoch == EPOCH:
            f,a = plt.subplots(2,view,figsize=(view,2))
            for i in range(view):
                img = np.reshape(sample_x.data.numpy()[i], (28,28))
                a[0][i].imshow(img, cmap='gray')
                a[0][i].set_xticks(()); a[0][i].set_yticks(())

            for i in range(view):
                img = np.reshape(sample_y.data.numpy()[i],(28,28))
                a[1][i].imshow(img, cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show()
    print('Training Ends!')

    print('Estimating Mutual Information...')

    IP(history_W1, history_b1, history_W2, history_b2, history_W31, history_b31, history_W32, history_b32, history_W4, history_b4,\
       history_W5, history_b5, history_W6, history_b6, history_eps, history_trainloss, history_testloss)























