# The Good Vibes Package for Vehicle Indentification from Acoustic and Seismic Information
Source code for xTech Good Vibrations Challenge.

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
$ pip3 install .
```
4. (Optional) If you want to use the developer features also install the developer requirements using:
```bash
$ pip3 install -r dev-requirements.txt
```

# Notebooks
We have supplied a couple useful notebooks listed below:
- **data_exploration.pynb**: This notebook is our first hack at visualizing the data. 