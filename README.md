# Enrollment Forecasting

Enrollment forcasting is a repository containing machine learning models to predict future enrollment of WWU computer science and computer science adjacent courses. 

# Installation

Conda Setup for Linux Terminal 

  ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest- Linux-x86_64.sh ```  
  (download conda package from internet)

  
  ```bash Miniconda3-latest-Linux-x86_64.sh```
  
  (run conda installation script)
  
  Press Enter, Scroll through the agreement, …, accept default installation path 
  ```source $your/path/to/miniconda$/etc/profile.d/conda.sh```  (tell the machine to know where conda is located) make sure to add own path
  ```conda --version```  (check for successful installation)
  Your steps to create an environment:
  Create environment from scratch
  ```conda create --name envname python==3.10  (create conda environment)```
  check python version that you’re currently using with python --version, choose own name for environment
  ```conda activate envname ```
  (activate the environment with name envname)
  ```pip install torch torchvision matplotlib scikit-learn pandas PyYAML wandb ```(install required python libraries into conda environment)
