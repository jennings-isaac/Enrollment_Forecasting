# Enrollment Forecasting

Enrollment forcasting is a repository containing machine learning models to predict future enrollment of WWU computer science and computer science adjacent courses. 


![image](https://github.com/user-attachments/assets/790a74aa-7c1c-4a39-995a-fb29214cec82)

## Installation

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
  `pip install torch torchvision matplotlib scikit-learn pandas PyYAML wandb`

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

## Training

This repo contains 2 machine learning models. One using sklearn MLP regressor and the other using a pytorch DNN. Both take in the same curated data, and predict the actual enrollment for a given class. To use new data from the WWU registrar, use the Client_Tutorial and plug in the new data. To change or upgrade the models or data curation use –INSERT NAME OF QUOCS NOTEBOOK–.

## Visualizations

There are visualizations that can be created using the Client_Tutorial, with the raw code in visualizations.py.

![image](https://github.com/user-attachments/assets/a866994a-a221-4b29-b945-9312f8507b65)
## Contacts

- [Conor Enfield](mailto:conore@live.com)
- [Isaac Jennings](mailto:jenningi2@wwu.edu)
- [Dang Hang Quoc Nguyen](mailto:dangn2@wwu.edu)
- [Piper Wolters](mailto:wolterp@wwu.edu)
