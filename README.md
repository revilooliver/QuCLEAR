# QuCLEAR  
**Artifact for the paper**: *QuCLEAR: Clifford Extraction and Absorption for Quantum Circuit Optimization*

---

## Installation

We recommend creating an **Anaconda virtual environment** before proceeding with the installation.  
[Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)

### **1. Clone the GitHub Repository**
First, clone the QuCLEAR repository and navigate to its directory:

```bash
git clone https://github.com/revilooliver/QuCLEAR.git
cd QuCLEAR
```

### **2. Install Required Packages**
Install the required Python packages via `pip`:

```bash
pip install -r requirements.txt
```

### **3. Environment Setup for Artifact Evaluation**
For artifact evaluation, we compare **QuCLEAR** with several other tools: **Rustiq**, **Qiskit**, **Paulihedral**, and **pytket**. To simplify the installation and environment management, we provide an automated script, `install_all.sh`, located in the `artifact_evaluation` folder:

```bash
cd artifact_evaluation
./install_all.sh
```

This script will create two Anaconda virtual environments:
- `QuCLEAR_env`: Contains Qiskit, Rustiq, and pytket.
- `PH_env`: Contains Paulihedral.

### **4. Test Installation**
We provide a simple test script to validate the installation. It will compile the circuits for Hamiltonian simulation:

```bash
./test_experiments_fast.sh
```

---

## Validating the Experiments

The experimental data can be validated by running the full experiment script:

```bash
cd artifact_evaluation
./test_experiments_full.sh
```

### **Results**
The compilation results are saved as JSON files in the `experiments/results_fullyconnected` folder.  

For example:  
- **`test_quclear_LiH.json`** contains results for **QuCLEAR**, **Qiskit**, and **Rustiq** for the LiH benchmark.  

### **Data Explanation**
- **`our_method`**: Results using QuCLEAR.
- **`combined_method`**: Results using QuCLEAR combined with Qiskit's optimization (reported in the paper).

---

## Generating Table Data

To generate the experimental tables in our paper, use the provided Jupyter notebooks:

1. **`generate_table3.ipynb`**: Generates data for Table 3.  
2. **`generate_table4.ipynb`**: Generates data for Table 4.  

These notebooks process results stored in the `experiments/results_fullyconnected` folder.

---

## Tutorials

We provide two example tutorials in the `tutorial` folder.

### **1. Optimizing Circuits and Calculating Expectation Values**
Run the `VQE_observables.ipynb` notebook to learn how to:  
- Optimize quantum circuits.  
- Calculate expectation values in variational quantum algorithms (e.g., VQE).

### **2. Absorbing CNOT Networks in QAOA**
Run the `QAOA_probabilities.ipynb` notebook to see examples of:  
- Absorbing CNOT networks.  
- Optimizing circuits for QAOA.