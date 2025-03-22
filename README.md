# Enrollment Forecasting

Enrollment forcasting is a repository containing machine learning models to predict future enrollment of WWU computer science and computer science adjacent courses. 


<img width="549" alt="image" src="https://github.com/user-attachments/assets/9c3e213f-0ae8-4543-bb37-b668fb98b471" />


## 🔵 Installation

### Conda Setup for Linux Terminal  

- **Download Conda package from the internet:**  
  `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`  

- **Run Conda installation script:**  
  `bash Miniconda3-latest-Linux-x86_64.sh`  

- **Follow installation steps:**  
  - Press **Enter** to continue.  
  - Hold enter through the agreement and type yes 
  - Accept the default installation path or choose a custom one.  

- **Tell the machine where Conda is located:**  
  `source $your/path/to/miniconda$/etc/profile.d/conda.sh`  
  *(Replace `$your/path/to/miniconda$` with the actual installation path.)*  

- **Verify installation:**  
  `conda --version`  

### Creating a Conda Environment  

- **Create an environment from scratch:**  
  `conda create --name envname python==3.10`  
  *(Replace `envname` with your preferred environment name. Check your current Python version using `python --version`.)*  

- **Activate the environment:**  
  `conda activate envname`  

- **Install required Python libraries:**  
  `pip install torch torchvision matplotlib scikit-learn pandas PyYAML wandb blob`

### Jupyter Notebook  

- **Install Jupyter Notebook:**  
  `pip install jupyter`  
  *(Installs the package that makes Jupyter Notebooks work.)*  

- **Create a new notebook (opens automatically):**  
  `jupyter notebook`  

- **Open an existing notebook (opens automatically):**  
  `jupyter notebook your_notebook.ipynb`  
  *(Replace `your_notebook.ipynb` with the actual notebook name.)*  

- **Important Notes:**  
  - Ensure that your Conda environment, created in the steps above, is **activated** before starting Jupyter Notebook.  
  - This ensures that the notebook has access to the installed packages.  
  - Place the notebook in the **root of your repository** so you can easily import submodules.


## 🔵 File Structure

### **Tutorials & Documentation**
- **`README.md`** – Main documentation file explaining the project.
- **`Inference_Tutorial.ipynb`** – Notebook guiding users on how to perform inference using the trained model.
- **`Tutorial.ipynb`** – General tutorial covering data processing, model training, and evaluation.

### **Model Files**
- **`best_model.pth`** – Saved PyTorch model checkpoint after training.
- **`sklearn_model.joblib`** – Trained scikit-learn model saved in joblib format.

### **Data Processing & Preparation**
- **`create_datasets.py`** – Script for creating datasets from raw data.
- **`curation.py`** – Script for cleaning and curating data before model training.
- **`dataset.py`** – Handles dataset loading, transformations, and visualization.
- **`enrollment_dataset.py`** – Dataset class for pytorch model.
- **`transform_data.sh`** – Shell script for automating data transformation tasks.

### **Model Training**
- **`pytorch_trainer.py`** – Defines the PyTorch model architecture.  
- **`pytorch_model.py`** – Script to train models using PyTorch.
- **`sklearn_trainer.py`** – Defines the scikit-learn model architecture.
- **`sklearn_model.py`** – Script for training scikit-learn models.
- **`early_stopping.py`** – Implements early stopping for PyTorch training.
- **`kfold_test.py`** – Script for performing k-fold cross-validation on the dataset.

### **Inference & Model Representation**
- **`model_representation.py`** – Visualizations using model data.
- **`visualizations.py`** – Script for generating visualizations of data and model performance.
- **`run_scripts.py`** – Automates running different scripts in sequence.
- **`run_scripts.sh`** – Shell script for executing predefined model training and inference tasks.

### **Miscellaneous**
- **`run_scripts.py`** – Main script for executing training or evaluation pipelines.
- **`run_scripts.sh`** – Shell script to execute predefined model training and inference tasks.




## 🔵 Training

We train two categories of models in this repository: **Sklearn** and **Pytorch**.  
Both use the **Model** and **Trainer** architecture.  

### Sklearn Models  
In the `sklearn_model`, there are one option:  
- **MLPRegressor**    

#### Sklearn MLPRegressor Hyperparameters  
```python
hidden_layer_sizes = (100,)
activation = 'tanh'
solver = 'adam'
learning_rate = 'adaptive'
max_iter = 1000
alpha = 0.0001
```

These hyperparameters were determined to be the best through **grid search**.  

### Pytorch Model  
In the `pytorch_model`, there is one option:  
- **RegressionDNN**
```
input_size= 86
hidden_sizes = [200, 200, 200, 200],
learning_rate=0.0001
batch_size=8,
num_epochs=500
patience=100,
```

### Training  
To train new models the required code is in the Tutorial.ipynb file.  
If using new data, ensure that:  
- It follows the required format.  
- It has been curated using our **curation file**.  

For a **detailed explanation** of our pipeline from raw data to training, refer to **`Tutorial.ipynb`**.  

## 🔵 Inference
To run inference using new data with a **pretrained model**, follow these steps:  

### Data Formatting  
First, the data must be in a csv formatted with these columns:  
- TERM
- CRN
- SUBJECT
- COURSE_NUMBER
- TITLE
- ACTUAL_ENROLL
- CAPENROLL
- PRIMARY_BEGIN_TIME
- PRIMARY_END_TIME
- U
- M
- T
- W
- R
- F
- S
- PRIMARY_INSTRUCTOR_TENURE_CODE
- CAMPUS

This data is assumed to be from the **WWU registrar**. Then, it will be run through curate to curate it to the format our model expects.

### Running Inference  
In `Inference_Tutorial.py`:  
1. Set the **path to the data**.  
2. Choose one of the **two pretrained models**:  
   - **Sklearn MLPRegressor**   
   - **Pytorch DNN**  
 

### Output  
Once the correctly formatted data is provided, the selected model will:  
- **Predict the actual enrollment** for the given classes.  
- **Output performance metrics** to evaluate the predictions.  


## 🔵 Visualizations

There are visualizations that can be created using the Inference_Tutorial, with the raw code in visualizations.py.

![image](https://github.com/user-attachments/assets/a866994a-a221-4b29-b945-9312f8507b65)
## 🔵 Contacts

- [Conor Enfield](mailto:conore@live.com)
- [Isaac Jennings](mailto:jenningi2@wwu.edu)
- [Dang Hang Quoc Nguyen](mailto:dangn2@wwu.edu)
- [Piper Wolters](mailto:wolterp@wwu.edu)
