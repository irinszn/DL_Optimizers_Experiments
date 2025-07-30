# Comparison of Optimizers under Noisy Data Conditions

This study is dedicated to the experimental analysis and comparison of the **accuracy and robustness** of three popular optimization algorithms — **SGD**, **Adam**, and **LAMB** — under noisy data conditions. The main goal of this work is to determine which optimization strategy (classical or adaptive) enables a **Convolutional Neural Network (CNN)** to better maintain its accuracy when trained on data with different types and levels of distortion.

During the experiments, the model was trained on the [**Animals-10**](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset, into which **Gaussian noise** and **Salt&Pepper noise** were artificially introduced at varying intensities. Model accuracy and performance were evaluated on a clean test set using multiple metrics, including **Accuracy**, **Precision**, **Recall**, **F1-score**, as well as the **convergence time** to a target loss level.

---

## 📁 Project Structure

The project consists of two main artifacts located in the `src/` directory:
```
src/
├── optimizers_with_noise_experiments.ipynb — A Jupyter Notebook containing the full code for running experiments, analysis, and result visualization.
└── report.pdf — A final report in PDF format presenting theoretical background, methodology description, result analysis, and conclusions.
```

## 🚀 How to Run the Experiments

To reproduce the results, follow the steps below. It is recommended to use **Google Colaboratory** with a **GPU accelerator** to significantly reduce computation time.

### 1. Environment Setup

- **Open the notebook**: Upload and open the `src/optimizers_with_noise_experiments.ipynb` file in Google Colab.
- **Enable GPU**: In the menu, go to `Runtime -> Change runtime type` and set the hardware accelerator to **GPU**.

### 2. Running the Notebook

Execute the Jupyter Notebook cells in order:

- **Install dependencies and import libraries**: The initial cells install all required packages (`torch-optimizer`, `seedbank`, etc.) and import libraries.
- **Download and prepare data**: This code block will automatically download the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset from Kaggle, unzip it, apply the necessary transformations, and split it into training, validation, and test sets.
- **Define models and helper functions**: These cells define the CNN architecture and helper functions for training, evaluation, and visualization.
- **Run experiments**: The main block will launch training loops for all noise scenarios and optimizers. This step is the most time-consuming and may take between **2 to 4 hours**.
- **Analyze and visualize results**: The final cells process the collected data, save performance metrics, and generate all relevant plots for analysis.

### 3. Viewing the Results

After completing the notebook, all graphs and result tables will be displayed in the output cells. A detailed analysis and conclusions are available in the file `src/report.pdf`.
