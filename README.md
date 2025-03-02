# Enrollment Forecasting

Enrollment forcasting is a repository containing machine learning models to predict future enrollment of WWU computer science and computer science adjacent courses. 

# Installation

## Conda Setup for Linux Terminal  

- **Download Conda package from the internet:**  
  `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`  

- **Run Conda installation script:**  
  `bash Miniconda3-latest-Linux-x86_64.sh`  

- **Follow installation steps:**  
  - Press **Enter** to continue.  
  - Scroll through the agreement and accept it.  
  - Accept the default installation path or choose a custom one.  

- **Tell the machine where Conda is located:**  
  `source $your/path/to/miniconda$/etc/profile.d/conda.sh`  
  *(Replace `$your/path/to/miniconda$` with the actual installation path.)*  

- **Verify installation:**  
  `conda --version`  

## Creating a Conda Environment  

- **Create an environment from scratch:**  
  `conda create --name envname python==3.10`  
  *(Replace `envname` with your preferred environment name. Check your current Python version using `python --version`.)*  

- **Activate the environment:**  
  `conda activate envname`  

- **Install required Python libraries:**  
  `pip install torch torchvision matplotlib scikit-learn pandas PyYAML wandb`  
