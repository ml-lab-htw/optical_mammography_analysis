from itertools import combinations
from tqdm import tqdm
from pathlib import Path
import logging
import pickle

homedir=str(Path( __file__).parents[1])  

def loadall(filename):
    """Iterativly loads lines of pickel file. 

        Paramaeters: 
           filename -- name of the pickel file to load
        Yield:
            line from pickel file
    """
    logger=logging.getLogger(f"helpers loadall()")
    try:
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    except Exception as err:
        logger.error(err)

