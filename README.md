# Comparison of Optimizers under Noisy Data Conditions

This study is dedicated to the experimental analysis and comparison of the **accuracy and robustness** of three popular optimization algorithms — **SGD**, **Adam**, and **LAMB** — under noisy data conditions. The main goal of this work is to determine which optimization strategy (classical or adaptive) enables a **Convolutional Neural Network (CNN)** to better maintain its accuracy when trained on data with different types and levels of distortion.

During the experiments, the model was trained on the [**Animals-10**](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset, into which **Gaussian noise** and **Salt&Pepper noise** were artificially introduced at varying intensities. Model accuracy and performance were evaluated on a clean test set using multiple metrics, including **Accuracy**, **Precision**, **Recall**, **F1-score**, as well as the **convergence time** to a target loss level.

---

## 📁 Project Structure

```
DL_Optimizers_Experiments/
│
├── experiment_notebooks/
│   ├── first_experiments.ipynb
│   ├── main_experiments.ipynb 
│   └── first_results_report.pdf
│
├── src/      # Framework code (in progress)
├── tests/
│
├── pyproject.toml
├── requirements.txt
└── requirements.dev.txt
```

## ⚠️ Results Status

`first_results_report.pdf` contains preliminary results from early experiment runs.

Full updated benchmarks are not yet available due to limited GPU resources required for large-scale training.

## 🚀 How to Run the Experiments

To reproduce the results, follow the steps below. It is recommended to use **Google Colaboratory** with a **GPU accelerator** to significantly reduce computation time.

### 1. Environment Setup

- **Open the notebook**: Upload and open the `experiment_notebooks/main_experiments.ipynb` file in Google Colab.
- **Enable GPU**: In the menu, go to `Runtime -> Change runtime type` and set the hardware accelerator to **GPU**.

### 2. Execution Pipeline

Run the notebook cells **in order**:

#### 1️⃣ Install dependencies  
Install required libraries and initialize the environment.

#### 2️⃣ Connect infrastructure  
- Connect to **MLflow** for experiment tracking  
- Mount virtual storage (e.g., Google Drive)

#### 3️⃣ Download dataset  
Download **Animals-10** from Kaggle and prepare raw data.

#### 4️⃣ Initialize model & noise modules  
- Create CNN model  
- Define noise generators (Gaussian, Salt & Pepper)

#### 5️⃣ Generate noisy datasets  
Create noisy dataset variants and save them to virtual storage.

#### 6️⃣ Load helper utilities  
Load training loops, evaluation functions, and logging utilities.

#### 7️⃣ Hyperparameter tuning (Optuna)  
Run hyperparameter optimization for selected optimizers.

#### 8️⃣ Run main experiment  
- Load configuration file  
- Execute training across optimizers and noise scenarios  
- Log results to MLflow  

---

### Configuration File (Example)

Before running the main experiment, load a YAML configuration file.

Example (simplified):

```yaml
mlflow:
  experiment_name: "SimpleCNN_Animals10"

data:
  dataset_name: "Animals10"
  num_classes: 10

model:
  name: "SimpleCNN"

training:
  epochs: 12
  batch_size: 64
  learning_rate: 0.001
  num_runs: 3

grid_search:
  optimizers:
    - name: "SGD"
      params:
        momentum: 0.9

  noise_scenarios:
    no_noise: []
    gaussian_0.05:
      - name: "GaussianNoiseAdder"
        params:
          std: 0.05
    salt_pepper_0.03:
      - name: "SaltAndPepperNoiseAdder"
        params:
          amount: 0.03
```


### 3. Viewing the Results

After execution:

- A results file with collected metrics is generated
- Summary tables are displayed in the notebook
- All runs are logged to MLflow
