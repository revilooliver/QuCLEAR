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

### **2. Environment Setup for Artifact Evaluation**

For artifact evaluation, **QuCLEAR** is compared with several tools: **Rustiq**, **Qiskit**, **Paulihedral**, and **pytket**. To simplify installation and environment management, we provide an automated script, `install_all.sh`, located in the `artifact_evaluation` folder:

```bash
cd artifact_evaluation
./install_all.sh
```

This script creates three Anaconda virtual environments:
- **`QuCLEAR_env`**: Contains QuCLEAR, Qiskit, and pytket.  
- **`PH_env`**: Contains Paulihedral.
- **`Tetris_env`**: Contains Tetris. 

#### **Install Rust and Rustiq in the `QuCLEAR_env`**

After setting up the environments, you will need to manually install **Rust** and **Rustiq** in the `QuCLEAR_env`.

1. **Activate the QuCLEAR environment**:
   ```bash
   conda activate QuCLEAR_env
   ```

2. **Install Rust**:  
   Follow the installation instructions provided on the official Rust website:  
   [Rust Installation Guide](https://www.rust-lang.org/tools/install)
   
   After installation, source your .bashrc file to enable Rust in your current session:
   ```bash
   source ~/.bashrc
   ```
   If you are working in a Conda environment, you may need to reactivate the environment to ensure the paths are properly configured:
      ```bash
   conda deactivate
   conda activate QuCLEAR_env
   ```

4. **Install Rustiq**:  
   Clone and install Rustiq from its GitHub repository:  
   [Rustiq Repository](https://github.com/smartiel/rustiq/tree/main)

Once these steps are complete, all required dependencies for artifact evaluation will be ready.


### **3. Test Installation**
We provide a simple test script to validate the installation. It will compile the circuits for Hamiltonian simulation:

```bash
./test_experiments_fast.sh
```

We also provide a simple test script to validate the experiments for mapping to hardware:
```bash
./test_mapping_experiments_fast.sh
```


---

## Validating the Experiments

The experimental data can be validated by running the full experiment scripts:

```bash
cd artifact_evaluation
./test_experiments_full.sh
./test_mapping_experiments_full.sh
```

### **Results**
The compilation results are saved as JSON files in the `experiments/results_fullyconnected` folder.  
The hardware mapping results are also saved as JSON files in the `experiments/results_ibm` and  `experiments/results_google` folder

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
3. **`generate_figure9.ipynb`**: Generates data for Figure 9.
4. **`generate_figure10.ipynb`**: Generates data for Figure 10.
5. **`generate_figure11_google.ipynb`** and **`generate_figure11_ibm.ipynb`**: Generates data for Figure 11.

These notebooks process results stored in the `experiments/results_fullyconnected` folder. The notebooks for figure11 process results in the `experiments/results_ibm` and  `experiments/results_google` folders.

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
