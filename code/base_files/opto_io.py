from base_files import opto_helpers as helpers
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path
import logging
import numpy as np
import pandas as pd

homedir=str(Path( __file__).parents[1]) 

def get_data(allPatients=True, preprocessing=False, six_features = False, two_features = False, ssd = False,tsd=False,val_per_bin = 399,length=400,pruning=False,pruning_path_left="",pruning_path_right=""):
    """Loading in the Matlab files,preprocessing and returning a dictonary of data, a list of names and the tumormap.

        Parmeters: 
            allPetients -- if set True loads and preprocesses all patients. Default is False where only clearly canceros and healty patients are loaded in
            preprocessing -- if set True apllys Preprocessing. Default is False
            six_features -- if set True adds six Features preprocessing. Default is False.
            two_features -- if set True adds two Features preproccessing. Default is False.
            ssd -- if set True adds Spatial Standard diviaten. Default is False.
            tsd -- if set True adds Temporal Standard diviation. Default is False.
            val_per_bin -- determens the amount of values per bin in Akkumulation preprocessing. Default = 399 wicht means no bins.
            length -- determens the timepoints used. Default = 400
            pruning -- if set True data pruning with pruning masks is applyed. 
            pruning_path_left -- sets the path for the pruning mask of the left breast.
            pruning_path_right -- sets the path for the pruning mask of the right breast. 

        Returns:
            data(dict): the dictonary containing a sorted version of the loaded, an potentily preprocessed data. use like: ["patient_id"]["side+wavelength"]["normalisation"] 
            tumor_map(dict): a dictonary containing information if and where a patient has a tumor, per patient
            X_names(list): a list of the patient ids 
    """
    logger=logging.getLogger(f"io get_data()")
    try:
        pruning_mask_left=[]
        if pruning_path_left !="":
            for e in helpers.loadall(pruning_path_left):
                pruning_mask_left=e
        
        pruning_mask_right=[]
        if pruning_path_right !="":
            for e in helpers.loadall(pruning_path_right):
                pruning_mask_right=e


        min_max_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        normalization_methods = {
            "minmax": min_max_scaler.fit_transform,
            "zscore": zscore,
            "robust": robust_scaler.fit_transform,
            "relative_change":relative_change,
            "zero_centering":zero_centering
        }

        df = pd.read_excel(f"{homedir}/data/excel_files/Read63_Anonymous.xlsx", skiprows=1)
        pat_map = dict(zip(df["name"],df["Pat State"]))     # 2 = tumor; 1= has somthing no cancer, 0= has nothing   
        tumor_map= dict(zip(df["name"],df["L/R tumor"]))    # 1 = tumor left or both; -1 = tumor right; if healthy still has val 1

        mypath = f"{homedir}/data/mat_files/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        main_data_types = ["left_wl1", "left_wl2", "right_wl1", "right_wl2"]
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        #load all dataset, normalise and preprocess them per patient 
        for dataset in tqdm(onlyfiles,desc="LOADING DATA",colour='blue'):
            patient_id = dataset.replace(".mat","")
            #skip all patients with unclear diagnoses
            if(not allPatients): 
                if pat_map[patient_id] ==1: 
                    continue
            matlab_data = loadmat(join(mypath, dataset))
            main_data = matlab_data["S"][0][0]
            #iterate over side an wavelength
            for index_main_data_type, name_main_data_type in enumerate(main_data_types):
                if pruning:
                    if(name_main_data_type.split("_")[0]=="left"):
                        timeseries_data = np.delete(main_data[index_main_data_type][0:length],pruning_mask_left,1)
                    else:
                        timeseries_data = np.delete(main_data[index_main_data_type][0:length],pruning_mask_right,1)
                else:
                    timeseries_data = main_data[index_main_data_type][0:length]
                #iterate over normalization methodes
                for norm_method in normalization_methods:
                    temp=normalization_methods[norm_method](timeseries_data)
                    if(preprocessing):
                        #sixFeatures preprocessing
                        if(six_features):
                            SMTSD = np.mean(np.std(temp,axis=0)) 
                            SSDTSD = np.std(np.std(temp,axis=0)) 
                            TMSSD = np.mean(np.std(temp,axis=1))
                            TSDSSD = np.std(np.std(temp,axis=1))
                            TMSM = np.mean(np.mean(temp,axis=1))
                            TSDSM = np.std(np.mean(temp,axis=1))
                            data[patient_id][name_main_data_type][norm_method]["sf"]= np.array([SMTSD,SSDTSD,TMSSD,TSDSSD,TMSM,TSDSM])
                        
                        
                        #Akkumulation preprocessing
                        tsd_val = np.std(temp,axis=0)
                        ssd_val = np.std(temp,axis=1)
                        if(two_features):
                            data[patient_id][name_main_data_type][norm_method]["tf"]= np.hstack((tsd_val,ssd_val)) 
                        if(tsd):
                            if(val_per_bin!=399):
                                bined=[]
                                for i in range(int(len(temp)/val_per_bin)):
                                    zws = np.std(np.array(temp[i*val_per_bin:(i+1)*val_per_bin]),axis=0)
                                    bined.append(zws)
                                data[patient_id][name_main_data_type][norm_method]["tsd"]=np.array(bined).flatten()
                            else:
                                data[patient_id][name_main_data_type][norm_method]["tsd"]=tsd_val
                        if(ssd):
                            data[patient_id][name_main_data_type][norm_method]["ssd"]=ssd_val
                        

                    temp=np.array(temp).flatten()    
                    data[patient_id][name_main_data_type][norm_method]["op"] = temp
        #creat the list of patient ids                    
        X_names =[ex for ex in data]
        X_names = np.array(X_names).reshape(-1)
        #cleaning of tumor_map caus: healty pat also have 1 in tumor_map
        for name in X_names:                                        
            if pat_map[name]==0 or pat_map[name]==1:
                tumor_map[name]=0
        return data,tumor_map,X_names
    except Exception as err:
        logger.error(err)



def relative_change(dataset):
    """normalizes the dataset bei dividing each value with the mean and returns a normalized dataset."""
    logger=logging.getLogger(f"io relative_change()")
    try:
        mean = np.mean(dataset)
        for row in dataset: 
            for i in range(len(row)):
                row[i] = (row[i]/mean)
        return dataset
    except Exception as err:
        logger.error(err)

def zero_centering(dataset):
    """normalizes the dataset bei dividing each value with the mean and returns a normalized dataset."""
    logger=logging.getLogger(f"io zero_centering()")
    try:
        mean = np.mean(dataset)
        for row in dataset: 
            for i in range(len(row)):
                row[i]=(row[i]-mean)
        return dataset
    except Exception as err:
        logger.error(err)
