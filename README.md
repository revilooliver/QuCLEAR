Here is the improved `README.md` in full Markdown format:

---

```markdown
# QuCLEAR  
**Artifact for the paper**: *QuCLEAR: Clifford Extraction and Absorption for Quantum Circuit Optimization*

---

## Installation  

We recommend creating an **Anaconda virtual environment** before installing QuCLEAR.  
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

### **3. Artifact Evaluation Environment Setup**  
QuCLEAR artifact evaluation compares **Rustiq**, **Qiskit**, **Paulihedral**, and **pytket**. To simplify environment setup, use the provided script `install_all.sh` in the `artifact_evaluation` folder:  

```bash
cd artifact_evaluation
./install_all.sh
```

This script creates two conda environments:  
- **`QuCLEAR_env`**: Contains Qiskit, Rustiq, and pytket.  
- **`PH_env`**: Contains Paulihedral.  

### **4. Test Installation**  
Validate the installation by running the provided test script. This script compiles circuits for Hamiltonian simulation:  

```bash
./test_experiments_fast.sh
```

---

## Validating the Experiments  

The experimental data can be validated by running the full experiment script in the `artifact_evaluation` folder:  

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
- **`combined_method`**: Results using QuCLEAR combined with Qiskit's optimization, as reported in the paper.

### **Generate Data for Paper Tables**  
To generate data for tables in the paper, use the provided Jupyter notebooks:  

1. **`generate_table3.ipynb`**: Generates data for Table 3.  
2. **`generate_table4.ipynb`**: Generates data for Table 4.  

These notebooks read data from the `experiments/results_fullyconnected` folder.

---

## Tutorials  

We provide two tutorial examples in the `tutorial` folder:  

### **1. VQE Observables**  
Run the `VQE_observables.ipynb` notebook to:  
- Learn how to optimize quantum circuits.  
- Calculate expectation values in variational quantum algorithms (e.g., VQE).  

### **2. QAOA Probabilities**  
Run the `QAOA_probabilities.ipynb` notebook to:  
- Absorb CNOT networks in QAOA.  
- Optimize circuits for QAOA execution.

---

## Summary  

- **Installation and Setup**: Instructions for installing and validating QuCLEAR and its environments.  
- **Artifact Evaluation**: Scripts for experiment validation and generating results.  
- **Tutorials**: Examples to help understand QuCLEAR's usage in quantum optimization and circuit compilation.

For any issues or questions, please feel free to connect us: ji.liu@anl.gov
```
