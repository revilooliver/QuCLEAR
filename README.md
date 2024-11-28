# QuCLEAR
Artifact for the QuCLEAR paper
## Installation
Creating a anaconda virtual environment is recommended before installing.
```
conda create --name QuCLEAR python=3.10
conda activate QuCLEAR
```
Then clone the github repo:
```
git clone https://github.com/revilooliver/QuCLEAR.git
cd QuCLEAR
```

Install the required packages via pip:
```
pip install -r requirements.txt
```
## Testing
Run the VQE_observables.ipynb notebook to find the examples of optimizing the circuits and calculating the expectation values
Run the QAOA_probabilities.ipynb notebook to find the examples of absorbing the CNOT network in QAOA. The benchmark_QuCLEAR_qiskit.ipynb contains the code for running more comparisons


