# Machine learning for Optical Mammography

The following repository provides the source code for the experiments presented in the paper "Detection of breast cancer using machine learning on time-series
diffuse optical transillumination data" by Nils Harnischmacher, Erik Rodner, and Christoph H. Schmitz.
We are applying machine learning algorithms to dynamic optical mammography data, enabling advanced analysis and insights.

Please note that the current documentation of the code is work-in-progress. 

## Installation / Development

The code is written in python and follows the classical workflow of installing python packages. Please note that
we tested all of the code on a Linux machine and there is no official support for Windows systems currently

1. **Create a virtual environment of your choice** - Python 3.10 is recommended
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Install our code as a package**
    ```bash
    cd code
    pip install .
    ```

## Obtaining the data

The data used in our study is provided at https://osf.io/4cr3z/. Please download the data and unzip it into the folder `code\data\downloaded_data`. After that, run the data preparation script to rename the files accordingly by: 
```bash
python code/data/data_preperation.py
```
If completed successfully, the files in the folder `code\data\downloaded_data` can be deleted.

## Usage

All experiments can be run individually. For example the classification results can be reproduced by:
```bash
python code/classification_moduls/classification.py
```

To switch between bilateral and unilateral modes, open the respective experiment's Python script and modify the settings at the bottom of the script.
**Example:**
```bash
if __name__ == "__main__":
    mode="uni" # change to "bi" here 
```
_Ensure you modify the mode (bilateral or unilateral) directly within the script before running._

## Experiment overview

1. ``code/classification_moduls/classification.py``- Experiment for bilateral and unilateral scenarious with a linear model
2. ``code/classification_moduls/autogluon.py`` - Experiments using AutoGluon (AutoML)
3. ``code/sparsification_simulation/sparsification.py``- Experiments using a reduced hardware setup (see Section 3.8 of our paper)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

If you use our code in our research, please cite our paper accordingly.