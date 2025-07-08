import glob, h5py, math, time, os, json, random, yaml, argparse, datetime
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm, expon, chi2, uniform, chisquare
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from numpy.random import choice
import pandas as pd
import numpy as np
import gc
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

from NNutils import *
from PLOTutils import *
from GENutils import *
parser   = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=False, default=None)
parser.add_argument('-s', '--seed', type=int, help="toy seed", required=False, default=None)
args     = parser.parse_args()

# random seed                                                                                                        
seed = args.seed
if seed==None:
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

# train on GPU?                                                                                                      
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# setup parameters
json_path = args.jsonfile
if json_path !=None:
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)

plot =  config_json["plot"]
# problem definition                                                                                                 
N_ref      = config_json["N_Ref"]
N_Bkg      = config_json["N_Bkg"]
N_Sig      = config_json["N_Sig"]
z_ratio    = config_json["w_ref"]
Pois_ON    = config_json["Pois_ON"]
print(N_Sig)
N_bkg_p, N_sig_p = N_Bkg,N_Sig
if Pois_ON:
    N_bkg_p = np.random.poisson(N_Bkg, size=(1,))[0]
    N_sig_p = np.random.poisson(N_Sig, size=(1,))[0]

signal_key = config_json["signal"]

##### define output path ######################                                                                      
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/seed%s/'%(seed)
folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
output_folder = folder_out

# DATA                                                                                                               
## ref and bkg                                                                                                       
data = []
for key in ['Wjj-3400', 'WWj-3400']:
    file = files[key]
    data.append(pd.read_csv(file,delim_whitespace = True,index_col =None).to_numpy())
data = np.concatenate(data, axis=0)
print(data.shape[0])
# draw from BKG data according to the weights
probability_distribution = data[:, -1]/np.sum(data[:, -1])
list_of_candidates_idx = np.arange(len(data))
number_of_items_to_pick = N_ref+N_bkg_p
draw = choice(list_of_candidates_idx, number_of_items_to_pick, replace=True,
              p=probability_distribution)
data = data[draw]
data = data[:, :-1] # remove weights from input data                                                                 
## sig 
data_s = []
for key in ['%s_HWlWh1-3400'%(signal_key), '%s_HWlWh2-3400'%(signal_key)]:
    file_s = files[key]
    data_s.append(pd.read_csv(file,delim_whitespace = True,index_col =None).to_numpy())
# draw from SIG data according to the weights
data_s = np.concatenate(data_s, axis=0)
probability_distribution = data_s[:, -1]/np.sum(data_s[:, -1])
list_of_candidates_idx = np.arange(len(data_s))
number_of_items_to_pick = N_sig_p
draw = choice(list_of_candidates_idx, number_of_items_to_pick, replace=True,
              p=probability_distribution)
data_s = data_s[draw]
data_s = data_s[:, :-1] # remove weights from input data                                                             

# concatenate datasets
feature = np.concatenate([data_s, data], axis=0)
target = np.concatenate([np.ones((N_sig_p+N_bkg_p, 1)), np.zeros((N_ref,1))], axis=0)
weights = np.concatenate([np.ones((N_sig_p+N_bkg_p, 1)), z_ratio*np.ones((N_ref,1))], axis=0)
print(feature.shape, target.shape)
print(np.mean(feature, axis=0), np.std(feature, axis=0))

# standardize                                                                                                        
feature = standardize(feature, mean_bkg, std_bkg)

# convert to torch tensors                                                                                           
feature = torch.from_numpy(feature).to(torch.float32)
target  = torch.from_numpy(target).to(torch.float32)
weights = torch.from_numpy(weights).to(torch.float32)

# model definition                                                                                                   
## gate                                                                                                              
gate_architecture = [16, 16, 5]
## experts                                                                                                           
lr = config_json["learning_rate"]
# specify which features are input to each expert
input_idx_matrix = [
    [0,1], # mWJJ, mJJ
    [2,3,4,5], # pT ordered
    [6,7,8,9], # mass ordered
    [10,11,12], # score 1 
    [13,14,15] # score 2
]

# specify architecture of each expert
exp_architecture = [
    [2,3,3,1], # mWJJ, mJJ                                                                                           
    [4,6,6,1], # pT ordered                                                                                          
    [4,6,6,1], # mass ordered                                                                                        
    [3,5,5,1], # score 1                                                                                             
    [3,5,5,1]  # score 2                                                                                             
]

L2_regularizer = config_json["L2_reg"]
patience     = config_json["patience"]
plt_patience = config_json["plt_patience"]
total_epochs = np.array(config_json["epochs"])
batch_size = int(feature.shape[0]/10)
accumulation_steps = int(target.shape[0]/batch_size)
n_experts = gate_architecture[-1]
    
train_data = CustomDataset(feature, target, weights)

# define the model
model = MixtureOfExperts_LocalGate(
    input_idx_matrix=input_idx_matrix,
    exp_architecture = exp_architecture,
    gate_architecture=gate_architecture,
    activation='sigmoid',
).to(DEVICE)

# initialize log arrays
# (they are filled at a rate defined by the patience)
loss_history      = np.array([])
val_loss_history  = np.array([])
epochs_history    = np.array([])
gate_history      = np.array([])

# Define loss function and optimizer                                                                                 
criterion = NPLMLoss
parameters = model.parameters()
print("Trainable parameters:")
print(count_parameters(model))
optimizer_adam = torch.optim.Adam(parameters, weight_decay=L2_regularizer)

# Initialize KFold
n_kfold_splits = config_json['n_kfold']
kfold = KFold(n_splits=n_kfold_splits, shuffle=True, random_state=42)

# Training loop                                                                                                      
t1 = time.time()
for epoch in range(total_epochs):
    # Split the dataset into k folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        # Create subsets for training and validation
        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        if not (epoch%patience): print(f"Fold {fold}:")        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=1)
        val_loader   = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=1)

        train_loss = train_loop(model, optimizer_adam, criterion,
                            train_loader, DEVICE,
                            accumulation_steps=accumulation_steps)

        val_loss = validation_loop(model, criterion,
                            val_loader, DEVICE,
                            accumulation_steps=accumulation_steps)
        if not (epoch%patience):
            with torch.no_grad():
                loss_history = np.append(loss_history, train_loss)
                val_loss_history = np.append(val_loss_history, val_loss)
                epochs_history = np.append(epochs_history, epoch)
                gate_tmp = gate_evaluation(model, criterion, val_loader, DEVICE,).cpu().detach().numpy().reshape((1, -1))
                if not len(gate_history): gate_history = gate_tmp
                else: gate_history = np.concatenate((gate_history, gate_tmp), axis=0)
            print('epoch: %i, loss: %f, val loss: %f'%(int(epoch+1),train_loss, val_loss))
            print('GATE COEFFS:', gate_history[-1, :])

        if torch.isnan(train_loss):
            del train_loss, optimizer
            break
        # check if skipping plots
        if plot==False: continue
        if ((epoch%plt_patience) or (epoch==0)) and (epoch!=(total_epochs-1)): continue
        # plots
        # loss evolution
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        plt.plot(epochs_history[2:], loss_history[2:], label='train loss')
        plt.plot(epochs_history[2:], val_loss_history[2:], label='val. loss')
        font=font_manager.FontProperties(family='serif', size=18)
        plt.legend(prop=font, loc='best')
        plt.ylabel('Loss', fontsize=18, fontname='serif')
        plt.xlabel('Epochs', fontsize=18, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.yticks(fontsize=16, fontname='serif')
        plt.grid()
        plt.savefig(output_folder+'loss.pdf')
        plt.show()
        plt.close()
        del fig, ax, font
        
        # evolution of the gate activation over time
        # (activation is averaged over the dataset)
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        for m in range(n_experts):
            plt.plot(epochs_history[:], gate_history[:, m], label='Gate %i'%(m))
        font=font_manager.FontProperties(family='serif', size=14)
        plt.legend(prop=font)
        plt.ylabel('Coeffs', fontsize=18, fontname='serif')
        plt.xlabel('Epochs', fontsize=18, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.yticks(fontsize=16, fontname='serif')
        plt.grid()
        plt.savefig(output_folder+'gate.pdf')
        plt.show()
        plt.close()
        del fig, ax, font

        # histogram of classifier output for the training data
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        with torch.no_grad():
            pred = model(feature.to(DEVICE))
            pred_norm = torch.sigmoid(pred[:, 0]).cpu().detach().numpy()
            print(pred_norm.shape)
        w = weights[:, 0]                                                                          
        bins = np.linspace(0., 1, 20)
        hD = plt.hist(pred_norm[target[:, 0]==1],
                      weights=w[target[:, 0]==1], bins=bins,
                      label='DATA', color='black',
                      lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(pred_norm[target[:, 0]==0],
                      weights=w[target[:, 0]==0],
                      color='#a6cee3', ec='#1f78b4', bins=bins,
                      lw=1, label='REFERENCE', zorder=1)
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0],
                     yerr= np.sqrt(hD[0]), color='black',
                     ls='', marker='o', ms=5, zorder=3)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
        font = font_manager.FontProperties(family='serif', size=18)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(0, 1)
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.xlabel("classifier output", fontsize=22, fontname='serif')
        plt.yscale('log')
        plt.savefig(output_folder+'out.pdf')
        plt.show()
        plt.close()
        del hD, hR, font, l, ax, fig

        # density ratio reconstruction projected on the various marginals
        w = w.detach().numpy()                                                                                               
        select_ref = 1*(target[:, 0]==0)                                                                                     
        select_dat = 1*(target[:, 0]==1)                                                                                     
        ref_preds = pred.cpu().detach().numpy()[select_ref>0]                                                                
        dat = feature.detach().numpy()                                                                                       
        for k in range(16):                                                                                                  
            dat_k = dat[:, k:k+1]                                                                                            
                                                                                                                         
            plot_reconstruction(data=dat_k[select_dat>0],                                                                    
                        weight_data=w[select_dat>0],                                                                     
                        ref=dat_k[select_ref>0],                                                                         
                        weight_ref=w[select_ref>0],                                                                      
                        ref_preds=[ref_preds],                                                                           
                        ref_preds_labels=['NPLM'],                                                                       
                        centroids=[],                                                                                    
                        t_obs=None, df=None,                                                                             
                        file_name='reco%i_epoch%i.pdf'%(k,epoch), save=True,                                             
                        save_path=output_folder,                                                                         
                        xlabels=[labels_dict[str(k)]], yrange={labels_dict[str(k)]: [-2, 2]},                            
                        bins=np.linspace(np.min(dat_k), np.max(dat_k), 30))   
t2=time.time()
print('End training')
print('execution time: ', t2-t1)

with torch.no_grad():
    pred = model(feature.to(DEVICE))
    nplm_loss_final = NPLMLoss(target.to(DEVICE), weights.to(DEVICE), pred).cpu().detach().numpy()
print('Final loss: ', nplm_loss_final)

# save test statistic                                                                                                
t_file=open(output_folder+'t.txt', 'w')
t_file.write("%f\n"%(-2*nplm_loss_final))
t_file.close()

# save exec time                                                                                                     
t_file=open(output_folder+'time.txt', 'w')
t_file.write("%f\n"%(t2-t1))
t_file.close()

# save monitoring metrics                                                                                            
np.save(output_folder+'loss_history', loss_history)
np.save(output_folder+'gate_history', gate_history)

 # save model                                                                        
checkpoint = {
    'weights': model.state_dict(),
    'optimizer': optimizer_adam.state_dict(),
}
torch.save(checkpoint,output_folder+'checkpoint')

print('Done')
