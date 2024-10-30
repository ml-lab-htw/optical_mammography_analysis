from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import numpy as np
import pandas as pd

homedir=str(Path.cwd().parents[0]) 
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')


def prep_data_autog(X,y):
    """ Converts the data into the specific Autogluon format 
     
        Paramaeters: 
            X -- List of the data matrixs 
            y -- Validation list
        Returns:
            df(pd.DataFrame): The Data Table including lable colum
            dfy(pd.DataFrame): The Validation list 
    """
    logger=logging.getLogger(f"ds_creation prep_data_autog()")
    try:
        dfx=pd.DataFrame(data=X)
        dfy=pd.DataFrame(data=y)
        dfy.rename(columns={0:"y"},inplace=True)
        df = pd.concat([dfx, dfy], axis=1)
        return df,dfy
    except Exception as err:
        logger.error(err)

def gen_eval_split(t_map,names):
    """ Generates the first data split into model selection data and testing data

        Paramaeters: 
            t_map -- a dictonary containing information if and where a patient has a tumor, per patient
            names -- list of the patient ids 
        Returns:
            Names_train(list): list of patient ids for the model selection split 
            Names_test(list): list of patient ids for the testing split 
    """
    logger=logging.getLogger(f"ds_creation gen_eval_split()")
    try: 
        X_temp=[]
        y_temp=[]
        for e in names:
            X_temp.append(e)
            y_temp.append(np.absolute(t_map[e]))
        Names_train, Names_test, y_train, y_test = train_test_split(X_temp,y_temp,test_size=0.2,random_state=0)
        return Names_train,Names_test
    except Exception as err:
        logger.error(err)



def getDataMatrix(data, patient_ids,tmap,name_main_data_types, norm_methods=["optonorm",]):
    """Creat bilateral data matrix and return X,y,id_list.

        Paramaeters: 
            data -- the preprocessed data that shall be convertet to the data matrix 
            patient_ids -- the list of patient ids 
            tmap -- the dictonary where the existenz of tumors is maped out
            name_main_data_types -- the list of witch dataset to use (left_wl1,left_wl2,right_wl1 or right_wl2)
            norm_methodes -- the lsit of witch normalization methods to use
        Returns:
            X(np.array): The data matrix 
            y(np.array): The Validation list 
            id_list(np.array): a referenz list for debuging, containing the patient and the wavelength 
    """
    logger=logging.getLogger(f"ds_creation getDataMatrix()")

    try:
        X_list = []
        y_list = []
        id_list = []
        #creat a vector for each patient containing the whished datasets and normalisations 
        for p in patient_ids:
            patient_data = data[p]
            patient_representations = []
            for nmdt in name_main_data_types:
                for nm in norm_methods:
                    patient_representations.append(patient_data[nmdt][nm])  
            x = np.hstack(patient_representations)
            X_list.append(x)
            y_list.append(abs(tmap[p]))
            id_list.append([p,nmdt])

        return np.vstack(X_list),np.array(y_list),np.array(id_list)
    except Exception as err:
        logger.error(err)



def getBreastBasedMatrix(data,patient_ids,tmap,wave_length=[1,],norm_methods=["zscore2",],preprocessing="op"):
    """Creat unilateral data matrix and return X,y,id_list.

        Paramaeters: 
            data -- the preprocessed data that shall be convertet to the data matrix 
            patient_ids -- the list of patient ids 
            tmap -- the dictonary where the existenz of tumors is maped out
            wave_length -- the list of witch wave_length to use (wl1 or wl2 or both)
            norm_methodes -- the lsit of witch normalization methods to use
        Returns:
            X(np.array): The data matrix 
            y(np.array): The Validation list 
    """
    logger=logging.getLogger(f"ds_creation getBreastBasedMatrix()")


    try:
        name_main_data_types=["left_wl","right_wl"]
        X_list = []
        y_list = []
        id_list = []
        #creat a vector for each breast containing the whished dataset based on wavelenght and normalization
        for p in patient_ids: 
            patient_data = data[p]
            for nmdt in name_main_data_types:
                patient_representations = []
                for nm in norm_methods: 
                    for w in wave_length:
                        patient_representations.append(patient_data[f"{nmdt}{w}"][nm][preprocessing])
                x=np.hstack(patient_representations)
                X_list.append(x)
                #creating the apropriet Validation list. So each breast get the right value for canceros or non canceros 
                if(tmap[p]==1 and nmdt.split("_")[0]=="left"):
                    y_list.append(1)
                elif(tmap[p]==-1 and nmdt.split("_")[0]=="right"):
                    y_list.append(1)
                else:
                    y_list.append(0)
                id_list.append([p,nmdt])

        
        return np.vstack(X_list),np.array(y_list)
    
    except Exception as err:
        logger.error(err)
    

def getDistanceDataMatrix(data, patient_ids,tmap,wave_length=[1,],norm_methods=["optonorm",],preprocess=False,preprocessing="op"):
    """Creat bilateral distance data matrix and return X,y,id_list.

        Paramaeters: 
            data -- the preprocessed data that shall be convertet to the data matrix 
            patient_ids -- the list of patient ids 
            tmap -- the dictonary where the existenz of tumors is maped out
            wave_length -- the list of witch wave_length to use (wl1 or wl2 or both)
            norm_methodes -- the lsit of witch normalization methods to use
        Returns:
            X(np.array): The data matrix 
            y(np.array): The Validation list 
    """
    logger=logging.getLogger(f"ds_creation getDistanceDataMatrix()")

    try:
        X_list = []
        y_list = []
        id_list = []
        #calculating the euclidic distance betwen the left and right breast for each patient with the whished wavelength and normalisations 
        for p in patient_ids:
            patient_data = data[p]
            patient_representations = []
            for nm in norm_methods:
                for w in wave_length: 
                    p1 = data[p][f"left_wl{w}"][nm][preprocessing]
                    p2 = data[p][f"right_wl{w}"][nm][preprocessing]
                    dist=np.linalg.norm(p1-p2)
                    patient_representations.append(dist)
            x = np.hstack(patient_representations)
            X_list.append(x)
            y_list.append(abs(tmap[p]))
            id_list.append([p,nm])
        if(preprocess):
            scaler=StandardScaler()
            scaler.fit(X_list)
            X_list=scaler.transform(X_list)
        return np.vstack(X_list),np.array(y_list)
    except Exception as err:
        logger.error(err)