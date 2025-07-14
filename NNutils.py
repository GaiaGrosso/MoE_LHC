import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from typing import List

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validation_loop(model, criterion, dataloader, device, accumulation_steps):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (data, target, weight) in enumerate(dataloader):
            data, target, weight = data.to(device), target.to(device), weight.to(device)
            # Forward pass
            output = model(data)
            loss = criterion(target, weight, output)
            total_loss += loss
            if (i + 1) % accumulation_steps == 0:
                total_loss = total_loss.item()
    return total_loss.cpu() / len(dataloader)


def gate_evaluation(model, criterion, dataloader, device):
    """
    returns the activation of each gate averaged over the given 
    dataset (accounting for events' weights)
    output shape: [n_experts,]
    """
    model.eval()
    gate_tmp = torch.zeros(model.n_experts)
    w_sum_tmp = 0
    with torch.no_grad():
        for i, (data, target, weight) in enumerate(dataloader):
            data= data.to(device)
            # Forward pass
            output = model.get_gate(data)
            gate_tmp+=torch.sum(output.cpu()*weight, dim=0)
            w_sum_tmp+=torch.sum(weight[:, 0])
    return gate_tmp/w_sum_tmp

def train_loop(model, optimizer, criterion, dataloader, device, accumulation_steps):
    model.train()
    total_loss = 0.0

    for i, (data, target, weight) in enumerate(dataloader):
        data, target, weight = data.to(device), target.to(device), weight.to(device)
        # Forward pass
        output = model(data)
        loss = criterion(target, weight, output)
        # Backpropagation with gradient accumulation
        total_loss += loss
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            # Logging or other operations can be done here
            total_loss = total_loss.item()
    return total_loss.detach().cpu() / len(dataloader)

class SimpleDNN(nn.Module):
    def __init__(self, architecture:List=[1,3,1], activation='relu'):
        super(SimpleDNN, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(architecture[i], architecture[i+1])
                       for i in range(len(architecture)-2)])
        self.output_layer = nn.Linear(architecture[-2], architecture[-1])
        self.activ = nn.ReLU()
        if activation=='sigmoid':
            self.activ = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activ(layer(x))
        x = self.output_layer(x)
        return x


class MixtureOfExperts_GlobalGate(nn.Module):
    """                                                                                                                 
    defines a list of experts DNN layers [e1(x), ..., en(x)]                                                            
    and a global gate with softmax activation used to perform dense mixture of experts                                  
    the initialization values for the expderts are given in lists                                                       
    the features input to each experts are given as argument [[i_11, i_12, ..., i_1m1], [i_21, i_22, ..., i_2m2], ...]  
    """
    def __init__(self, input_idx_matrix: List=[],
                 exp_architecture: List=[],
                 gate_coeffs: List=[],
                 gate_train_coeffs:bool=True,
                 activation='relu',
                 name=None, **kwargs):
        super(MixtureOfExperts_GlobalGate, self).__init__()

        if gate_train_coeffs:
            self.n_experts = len(exp_architecture)
            self.experts = nn.ModuleList([SimpleDNN(architecture=exp_architecture[i], activation=activation) for i in range(self.n_experts)])
            self.gate = nn.Parameter(gate_coeffs.reshape((self.n_experts,1)).type(torch.float32),
                                     requires_grad=gate_train_coeffs)
            self.input_idxs = input_idx_matrix
        else:
            experts = []
            gate_coeffs_squeezed = []
            input_idx_matrix_squeezed = []
            i=0
            for coeff in gate_coeffs:
                if coeff>0:
                    experts.append(SimpleDNN(architecture=exp_architecture[i]))
                    gate_coeffs_squeezed.append(coeff)
                    input_idx_matrix_squeezed.append(input_idx_matrix[i])
                i+=1
            self.experts = nn.ModuleList(experts)
            self.n_experts = len(experts)
            self.gate = nn.Parameter(torch.tensor(gate_coeffs_squeezed).reshape((self.n_experts,1)).type(torch.float32),
	                       requires_grad=gate_train_coeffs)
            self.input_idxs = input_idx_matrix_squeezed

    def forward(self, x):
        p = self.get_gate_coeffs() # [N_exp, 1]                                                                         
        y = []
        for j in range(self.n_experts):
            y.append(self.experts[j](x[:, self.input_idxs[j]]))
        y = torch.cat(y, dim=1) #[N, N_exp]                                                                             
        return torch.tensordot(y,p, dims=([1],[0])) # [N, 1]
    
    def get_gate_coeffs(self):
        return torch.nn.functional.softmax(self.gate, dim=0)

    def eval_expert_j(self, x, j):
        with torch.no_grad():
            return self.experts[j](x[:, self.input_idxs[j]])
    
class MixtureOfExperts_LocalGate(nn.Module):
    """
    defines a list of experts DNN layers [e1(x), ..., en(x)]
    and a local gate, g(x), with softmax activation used to perform dense mixture of experts
    the initialization values for the expderts are given in lists
    the features input to each experts are given as argument [[i_11, i_12, ..., i_1m1], [i_21, i_22, ..., i_2m2], ...] 
    """
    def __init__(self, input_idx_matrix: List=[],
                 exp_architecture: List=[],
                 gate_architecture: List=[],
                 activation='relu',
                 name=None, **kwargs):
        super(MixtureOfExperts_LocalGate, self).__init__()
        self.n_experts = len(exp_architecture)
        self.experts = nn.ModuleList([SimpleDNN(architecture=exp_architecture[i], activation=activation) for i in range(self.n_experts)])
        self.gate = SimpleDNN(gate_architecture, activation=activation)
        self.input_idxs = input_idx_matrix
        
    def forward(self, x):
        p = self.get_gate(x) # [N, N_exp]
        y = []
        for j in range(self.n_experts):
            y.append(self.experts[j](x[:, self.input_idxs[j]]))
        y = torch.cat(y, dim=1) #[N, N_exp]
        y = torch.multiply(y,p) # [N, N_exp]
        return torch.sum(y, dim=1, keepdim=True) #[N, 1]

    def get_gate(self, x):
        p = self.gate(x) # [N, N_exp]
        p = torch.nn.functional.softmax(p, dim=1) # [N, N_exp]
        return p

    def eval_expert_j(self, x, j):
        with torch.no_grad():
            return self.experts[j](x[:, self.input_idxs[j]])

        
class KernelMethod(nn.Module):
    '''                                                                                                   
    return: coeff * K(-0.5(x-mu)**2/scale**2)                                                                               
    '''
    def __init__(self, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod, self).__init__()
        self.positive_coeffs=positive_coeffs
        if self.positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        self.coeffs = Variable(coeffs.reshape((-1, 1)).type(torch.float32),
                               requires_grad=train_coeffs) # [M, 1]                                                                                               
        self.kernel_layer = KernelLayer(centroids=centroids, widths=widths,
                                        train_centroids=train_centroids, train_widths=train_widths,
                                        resolution_const=resolution_const, resolution_scale=resolution_scale,
                                        name='kernel_layer')


    def call(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]                                                                                                               
        W_x = self.coeffs  # [M, 1 ]                                                                                                                              
        out = torch.tensordot(K_x, W_x, dims=([1], [0]))
        return out

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def set_widths(self, widths):
        self.kernel_layer.set_widths(widths)
        return
        
    def set_width(self, width):
        self.kernel_layer.set_width(width)
        return
        
    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()
        return
        
    def clip_coeffs(self):
        self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)
        return

class KernelMethod_SoftMax_2(nn.Module):
    '''                                                                                                                                                          
    return: coeff * K(-0.5(x-mu)**2/scale**2) * softmax( -0.5(x-mu)**2/scale**2 )                                                                
    '''
    def __init__(self, centroids, widths, coeffs, resolution_const=0, resolution_scale=1, coeffs_clip=None,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod_SoftMax_2, self).__init__()
        self.train_coeffs=train_coeffs
        self.coeffs=coeffs
        self.epsilon=1e-10
        if not coeff_clip==None:
            if positive_coeffs:
                self.cmin=0
                self.cmax=coeffs_clip
            else:
                self.cmin=-coeffs_clip
                self.cmax=coeffs_clip
        self.coeffs = Variable(self.coeffs.reshape((-1, 1)).type(torch.float32),
                               requires_grad=train_coeffs) # [M, 1]                                                                                               
        self.kernel_layer = KernelLayer(centroids=centroids, widths=widths,
                                        train_centroids=train_centroids, train_widths=train_widths,
                                        resolution_const=resolution_const, resolution_scale=resolution_scale,
                                        name='kernel_layer')

    def call(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]                                                                                                              
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n, 1]                                                                                           
        out = torch.tensordot(torch.mul(K_x,K_x), self.coeffs, dims=([1], [0])) # [n, 1]
        out = torch.divide(out, Z) # [n, 1]                                                                                                                      
        return out

    def clip_coeffs(self):
        self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()

    def set_widths(self, widths):
        self.kernel_layer.set_widths(widths)
        return

    def set_width(self, width):
        self.kernel_layer.set_width(width)
        return


class KernelLayer(nn.Module):
    def __init__(self, centroids, widths, resolution_const=0, resolution_scale=1,
                 beta=None,
                 cmin=None, cmax=None,train_centroids=False, train_widths=True,
                 name=None, **kwargs):
        super(KernelLayer, self).__init__()
        self.resolution_const=resolution_const
        self.resolution_scale=resolution_scale
        self.cmin=cmin
        self.cmax=cmax
        self.beta=beta
        self.M = centroids.shape[0]
        self.d = centroids.shape[1]
        self.width = widths[0]
        self.centroids = Variable(centroids.type(torch.float32), requires_grad=train_centroids)
        self.cov_diag = self.width**2

    def call(self, x):
        if self.beta==None:
            out, arg = self.Kernel(x)
            return out, arg
        else:
            out, arg, out2, arg2 = self.Kernel(x)
            return out, arg, out2, arg

    def transform_widths(self):# transform width variable to account for resolution boudnaries (quadrature sum)                                                  
        widths = torch.add(self.widths**2, self.resolution_const**2) # [M, d]                                                                           
        widths+= torch.multiply(self.centroids, self.resolution_scale)**2 # [M, d]                                                                     
        widths = torch.sqrt(widths) # [M, d]                                                                                                                    
        return widths

    def get_widths(self):
        return self.width*torch.ones((self.M, self.d))

    def set_widths(self, widths):
        self.widths.data = widths
        self.compute_cov_diag()
        return

    def set_width(self, width):
        self.width = width
        self.compute_cov_diag()
        return

    def get_centroids(self):
        return self.centroids #[M, d]                                                                                                                             

    def clip_centroids(self):
        if (not self.cmin==None) and (not self.cmax==None):
            self.centroids.data = self.centroids.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        """                                                                                                                                                      
        sum_j(sum_i(K_i(mu_j))*log(sum_i(K_i(mu_j))))                                                                                                            
        return: scalar                                                                                                                                           
        """
        K_mu, _ = self.call(self.centroids) #[M, M]                                                                                                              
        K_mu = torch.mean(K_mu, axis=1) # [M,]                                                                                                                   
        entropy = torch.sum(torch.multiply(K_mu, torch.log(K_mu)))
        return entropy

    def compute_cov_diag(self):
        self.cov_diag = self.width**2
        return

    def gauss_const(self, cov_diag):
        """                                                                                                                                                      
        # widths.shape = [M, d]                                                                                                                                  
        Returns the normalization constant for a gaussian                                                                                                        
        # return.shape = [M,]                                                                                                                                    
        """
        det_sigma_sq = torch.sum(cov_diag, axis=1)# [M,]                                                                                                         
        return torch.sqrt(det_sigma_sq)/torch.pow(torch.sqrt(torch.tensor(2*torch.pi)), cov_diag.shape[1])

    def Kernel(self,x):
        """                                                                                                                                                      
        # x.shape = [N, d]                                                                                                                                       
        # widths.shape = [M, d]                                                                                                                                  
        # centroids.shape = [M, d]                                                                                                                               
        Returns the gaussian function exponent term                                                                                                              
        # return.shape = [N,M]                                                                                                                                   
        """
        dist_sq  = torch.subtract(x[:, None, :], self.centroids[None, :, :])**2 # [N, M, d]                                                                      
        arg = -0.5*torch.sum(dist_sq/self.cov_diag,axis=2) # [N, M]
        kernel = torch.exp(arg)
        if self.beta!=None:
            arg2 = -1*self.beta*torch.sum(dist_sq, axis=2) #[N, M]                                                                                               
            kernel2 = torch.exp(arg2)
            return kernel, arg, kernel2, arg2
        else:
            return kernel, arg # [N, M]



# losses
def NPLMLoss(true, weight, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = weight[:, 0]
    return torch.sum((1-y)*w*(torch.exp(f)-1) - y*w*(f))

def MSELoss(true, pred):
    f   = 1./(1+torch.exp(-1*pred[:, 0])) # sigmoid                                                                                                              
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum(((f-y)**2)*w)

def BCELoss(true, pred):
    f   = 1./(1+torch.exp(-1*pred[:, 0])) # sigmoid                                                                                                              
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum(-1*w*((1-y)*torch.log(1-f)+y*torch.log(f)))

def L2Regularizer(pred):
    return torch.sum(torch.multiply(pred,pred))

def L1Regularizer(pred):
    return torch.sum(torch.abs(pred))

def CentroidsEntropyRegularizer(entropy):
    return entropy
