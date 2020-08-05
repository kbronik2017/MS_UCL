[![GitHub issues](https://img.shields.io/github/issues/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/issues)
[![GitHub forks](https://img.shields.io/github/forks/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/network)
[![GitHub stars](https://img.shields.io/github/stars/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/stargazers)
[![GitHub license](https://img.shields.io/github/license/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/blob/master/LICENSE)


# Deep Multi-task Learning Framework for Image Segmentation (Cross Sectional and Longitudinal data analysis)


![multiple sclerosis (MS) lesion segmentation](images/brain_lesion.gif)


# Running the GUI Program! 

First, user needs to install Anaconda https://www.anaconda.com/

Then


```sh
  - conda env create -f cnn_run_conda_environment_cpu.yml  
``` 
and 

```sh
  - conda activate idptfcpu  
``` 
finally

```sh
  - python  CNN_Longitudinal_CrossSectional_GUI.py 
``` 

After lunching the graphical user interface, user will need to provide necessary information to start training/testing as follows:  

<br>
 <img height="510" src="images/help.png" />
</br>


# Running the Program from the command line!

First 

```sh
  - conda activate idptfcpu  
``` 
then for training


```sh
  - python  -m tbb training_script_Cross_Sectional.py  [or training_script_Longitudinal.py]
``` 

for testing

```sh
  - python  -m tbb inference_script_Cross_Sectional.py  [or inference_script_Longitudinal.py]
``` 

# Testing the Program!



