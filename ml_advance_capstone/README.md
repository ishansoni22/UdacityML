# Machine Learning Engineer Nanodegree
## Capstone Project

This github repository contains my final project for Udacity's Machine Learning Advanced Nanodegree.

This repository contains the following files/ directories :

1. *proposal.pdf* : The Machine Learning Engineer Nanodegree Capstone project proposal.
2. *data Folder* : This folder contains the data files (train + test) required for this capstone project.
3. *notebook.ipynb* : The jupyter notebook that contains the code for running the project.
4. *report.pdf* : The capstone project report addressing the five major project development stages.
5. *README.md* : The current file. It contains description about the capstone project and how to set it up locally.

## Project Setup
#### Kaggle : 
Since my project idea is taken from Kaggle, I've created a kernel on the Kaggle platform to help users avoid setting up the project on their local machines. Link to the kernel : 

#### Local Setup (Mac OS):
For a complete tutorial on how to setup a data science environment on your local mac machine, follow :
[Setup a data science environment on your mac guide](https://medium.com/@arunponnusamy/setting-up-deep-learning-environment-the-easy-way-on-macos-high-sierra-f1b6331ffc40)

1. Install HomeBrew. I used HomeBrew as a package manager to get all of my dependencies.
2. Install python.
```
brew install python3
```
3. Install Python's package manager pip.
```
brew install pip3
```
4. Install Virtual Environment. Virtual environment is a super useful tool to keep things clean and separate. All the Python packages you install will be virtually contained within a particular virtual environment you create and will not mess with the things outside. 
```
pip3 install virtualenv virtualenvwrapper
```
5. Update your bash settings. Add the following to your .bash_profile
```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
6. Create a Virtual Environment.
```
mkvirtualenv <env-name> -p python3
```
7. Activate the virtual environment.
```
workon <env-name>.
```
8. Install the necessary dependencies.
```
pip install numpy matplotlib scipy scikit-learn seaborn jupyter
```
9. Start up jupyter notebook.
```
jupyter notebook
```












