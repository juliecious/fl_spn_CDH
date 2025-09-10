
## How to Run

- Installation: R package.
  - Our federated conditional independent test method is developed based on [R Package](https://github.com/ericstrobl/RCIT), please follow their procedures and install all the dependencies at first.

- Installation: Environment.
```sh
# Set up a new conda environment with Python 3.8.
conda create -n FedCDH python=3.10
conda activate FedCDH

# Please navigate to the root directory, and install our source code.
pip install -e .

# Install other python libraries.
pip install causaldag rpy2 numpy scipy tqdm networkx
```

- Evaluation: quick start.
```sh
# Parameters:
#     N: number of instances to evaluate
#     d: number of variables
#     K: number of clients
#     n: number of samples in one client
#     model: data generation model, linear or general.
cd tests
python TestFedCDH.py --N 10 --d 6 --K 10 --n 100 --model linear
```
