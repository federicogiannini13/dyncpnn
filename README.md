# DYNcPNN
This repository accompanies the paper "Dynamic Continuous Progressive Neural Networks for Evolving Streaming Time Series" by Federico Giannini, Giacomo Ziffer, and Emanuele Della Valle (Politecnico di Milano).

It contains the full implementation of the proposed DYNcPNN model and the code used to run the experiments presented in the paper.

## 1) Installation
execute:

`conda create -n env python=3.8`

`conda activate env`

`pip install -r requirements.txt`

## 2) Project structure
The project is composed of the following directories.
#### datasets
Download `datasets.zip` from [here](https://drive.google.com/file/d/12MjLQhkL-EAS1dd0RlxP0bJ5rvmHG6j5/view?usp=sharing).
It contains the configurations of each data sources.
Each file's name has the following structure: **\<data_source\>\_\<id_configuration\>conf\_\<train_or_test\>.csv**.

<ins>Data sources:</ins>
* air_quality: AirQuality.
* energy: PowerConsumption.
* sine_rw10_mode5: SRWM
* weather: Weather

<ins>Train or test:</ins>
* train: The data stream contains the data points for the prequential evaluation.
* test: The data stream contains the data points of the test sets for the CL evaluation. Each concept (task column) is represented by 2k data points.

#### models
It contains the python modules implementing DYNcPNN, cPNN, cLSTM and ARF$_T$.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### data_gen
It contains the python modules implementing the data stream generator.
#### detectors
It contains the python modules implementing the DYNcPNN's Sentinel.

## 3) Evaluation
#### evaluation/test.py
It runs the prequential evaluation and CL eveluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py** and **evaluation/cl_evaluation.py**


## 3) Detectors
#### evaluation/sentinel.py
It implements the DYNcPNN's Sentinel.

#### evaluation/sentinel_simulator.py
It simulates a Sentinel with desired recall and precision values.
