import os, json, argparse, glob, time, datetime
import numpy as np
import os.path

# specify here where you want to save the output of your training
OUTPUT_DIRECTORY =  './out/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary
config_json = {
    "N_Ref"   : 17700,#number of bkg events in class 0
    "N_Bkg"   : 1770, #number of bkg events in class 1
    "N_Sig"   : 36*2, #number of signal events in class 1
    "signal" : 'Zp3500-Hc800', # it loops over sasmples ['Zp3500-Hc800_HWlWh1-3400', 'Zp3500-Hc800_HWlWh2-3400'],
    "w_ref": 0.1, #luminosity ratio between class 1 and class 0 
    "output_directory": OUTPUT_DIRECTORY,
    "epochs": 10,
    "patience": 1, #interval between saved monitored metrics
    "plt_patience": 10, #interval between new plots generation (if plot==True)
    "plot": True, #whether to make plots
    "learning_rate" : 0.001,
    "n_kfold": 2, # number of folds in the k-fold procedure
    #"gate_coeffs": [1. for _ in range(5)], # this is used as a mask for which experts to train on 
    #"gate_coeffs": [1, 0, 0, 0, 0],
    #"gate_train_coeffs" : True,
    "L2_reg": 0.1,
    "Pois_ON": True, # whether data should have poissonian number of events
}

# training specs
gate_str = ''
#if config_json["gate_train_coeffs"]:
#    gate_str += 'trainGate'
#else:
#    gate_str += 'frozenGate'
#for m in config_json["gate_coeffs"]:
#    gate_str += '-%s'%(str(m))
ID = gate_str
ID += "L2reg%s"%(str(config_json["L2_reg"]))
ID += '_epochs%s'%(config_json["epochs"])
ID += '_kfold%i'%(config_json["n_kfold"])
ID += '_lr%s'%(str(config_json["learning_rate"]))
if "gather_after" in list(config_json.keys()):
    ID+='_gathergrad%i'%(config_json["gather_after"])
if not config_json["Pois_ON"]:
    ID += '_NO-Pois'

# problem specs                                                                                                                                      
ID +='/Nref'+str(config_json["N_Ref"])
ID +='_Nbkg'+str(config_json["N_Bkg"])
ID +='_Nsig'+str(config_json["N_Sig"])
if config_json["N_Sig"]:
    ID += config_json["signal"]

config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+ID

#### launch python script ###########################                                                                                                
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=100)
    parser.add_argument('-s', '--firstseed', type=int, help="first seed for toys (if specified the the toys are launched with deterministic seed incresing of one unit)", required=False, default=-1)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    firstseed= args.firstseed
    config_json['pyscript'] = pyscript
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript_str+'/'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])

    json_path = create_config_file(config_json, config_json["output_directory"])
    
    if args.local:
        # launch the script locally
        if firstseed<0:
            seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            os.system("python %s/%s -j %s -s %i"%(os.getcwd(), pyscript, json_path, seed))
        else:
            os.system("python %s/%s -j %s -s %i"%(os.getcwd(), pyscript, json_path, firstseed))
    else:
        # submit the script via slurm job
        # (this is useful if you have access to computing resources in a cluster)
        label = "logs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):
            if firstseed>=0:
                seed=i
                seed+=firstseed
            else:
                seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            script_sbatch = open("%s/submit_%i.sh" %(label, seed) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-3:00\n")
            script_sbatch.write("#SBATCH -p iaifi_gpu_priority\n")
            #script_sbatch.write("#SBATCH -p gpu\n")
            #script_sbatch.write("#SBATCH -p serial_re\n")                                                                           
            script_sbatch.write("#SBATCH --mem=5000\n")
            script_sbatch.write("#SBATCH -o ./logs/%s"%(pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e ./logs/%s"%(pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
            script_sbatch.write("module load python/3.10.9-fasrc01\n")
            script_sbatch.write("module load cuda/11.8.0-fasrc01\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s/%s -j %s -s %i\n"%(os.getcwd(), pyscript, json_path, seed))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, seed))
            os.system("sbatch %s/submit_%i.sh"%(label, seed) )

            
