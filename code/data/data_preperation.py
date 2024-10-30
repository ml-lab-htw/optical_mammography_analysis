from base_files import opto_helpers as helpers
from scipy.io import loadmat,savemat
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
import os

homedir=str(Path( __file__).parents[1]) 
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')

def main():
    """ renames the downloaded .mat files 
    """
    path=f"{homedir}/data/mat_files/"
    df = pd.read_csv(f"{homedir}/data/map.csv",sep=";",)
    mypath = f"{homedir}/data/downloaded_data"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    if not os.path.exists(path):
        os.mkdir(path)
    for dataset in tqdm(onlyfiles,desc="PREPARING DATA",colour='blue'):
        matlab_data = loadmat(join(mypath, dataset))
        main_data = matlab_data["S"]
        name=main_data[0][0][7][0][0][0][0][0].strip().replace("'", "").split("=")[1].split("_")[0].replace("'", "")
        savemat(f"{path}{name}.mat",matlab_data)
    
if __name__ == "__main__":
    main()