# The Good Vibes Package for Vehicle Indentification from Acoustic and Seismic Information
Source code for xTech Good Vibrations Challenge.

# Notes to Reviewer(s)
Our solution is much more basic than previously proposed. We noticed our original approach was completely flawed and chose to just build you a simple detector and classifier based on a spectral principal component analysis and a few simple deep learning models that can run on an low compute/low power device. 

**Note:** We recognize that this solution was not what was proposed and if that means we are disqualified, we own that and apologize.

However, we believe (hope) we did come up with a useable methodology that (with more time/effort) could be built into a fuller solution and leverage our research in Bayesian Knowledge Bases (BKBs) as originally proposed. We would love to continue this effort with you, but we'll need the whole summer. 

## Available Predicted Features
1. **Detection:** We predict detection of a vehicle using our approach.
2. **Target ID:** If a vehicle was detected we try to predict it's identifier, i.e., exactly which target it is.

We realize this is just two features, but we think it is an important proof-of-concept as in our tests we can predict a vehicle in an acoustic/seismic signal with ~80% accuracy, and its true vehicle class with about ~75% accuracy. 

## Lessons Learned
### Zero Order Features
The data did not really enable prediction of zero order features such as fundemental harmonics. We attempted to conduct this analysis but with the noise present and limited sample size, we could not determine any meaningful results.

### First Order Features
While we pretected detection, we could not predict any target trajectories or speed as their was simply not enough meaningful data. All vehicles were approaching a approximately the same speed and their was also just not enough vehicles present. Further, given that output/prediction time only allows for a singal sensor to be used, we couldn't perform any Time-Distance-OF-Arrval (TODA) analysis. However, even when analyzing TODA for the experimental setups included in the dataset, we could only conduct a meaningful analysis on one location.

### Second Order Features
There wasn't enough data to get any reasonable weight class predictions as not enough different vehicles were represented. 

## Final Thoughts 
I believe with a more robust experimental design during data collection, as well as a effort to annotate these datasets in a more meaningful way, e.g., with a vehicle ontology, we could perform better analysis and even be able to conduct our Bayesian approach. We really believe the data is just too limited.

# Getting Started
This project is using Python 3.11.7 and should be compatible with most versions of Python, but we recommend using Python 3.11. Further, we will assume a *Linux* distribution such as Ubuntu. 

## Installing the project in virtual environments
1. Clone this repository, and navigate to your local path:
```bash
$ git clone https://github.com/yakaboskic/goodvibes.git && cd goodvibes
```
2. Install a Python 3.11 virtual environment in the project
```bash
$ python3.11 -m venv goodvibes-venv
```
3. Activate virtual environment:
```bash
$ . goodvibes-venv/bin/activate
```
3. Install the project into virtual environment:
```bash
$ pip3 install -e .
```
4. (Optional) If you want to use the developer features also install the developer requirements using:
```bash
$ pip3 install -r dev-requirements.txt
```

## Running Our Algorithm
To run our algorithm, simply run:
```bash
$ python3 scripts/run.py path_to_acoustic_signal.wav path_to_seismic_signal.wav output.csv
```
The program runs prediction over a 1 second interval, so processing a large audio file (such as a five minute file) will take some time. Our performance is generally less than a second for 3each one second interval. 

# Remarks and Notes
In this section we describe a number of complications that have arose throughout development. 

## Inconsistent Data Labeling
1. No wav data for the corresponding Eglin XML file: Node81-Ch1-21-2023_03_16_13_30_00-9866Hz.wav.xml
1. Solicitation Dataset 1 does not have consistent dates for Node 124, and therefore was not considered. 
1. Node 129 in BRFP also had inconsistent dates, changed signatures on 4-10-2023 to 4-12-2023. 
