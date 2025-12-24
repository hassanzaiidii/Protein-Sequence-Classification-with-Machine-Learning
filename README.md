# Project Title: Autophagy Protein Classification using Machine Learning

# Project Overview

The **Autophagy Protein Classification using Machine Learning** project focuses on the computational identification of autophagy-related proteins using protein sequence data. Autophagy is a vital biological process involved in cellular maintenance and disease regulation. The primary objective of this project is to develop a reliable binary classification system that can accurately distinguish **autophagy proteins** from **non-autophagy proteins** using machine learning techniques. Such predictive systems can assist biological research and reduce the dependency on time-consuming laboratory experiments.

# Dataset

The project utilizes a publicly available protein sequence dataset obtained from a GitHub repository. The dataset is designed specifically for the binary classification of autophagy-related proteins and is provided in FASTA format.

* **Total protein sequences**: 8,000
* **Training set**: 6,667 sequences
* **Testing set**: 1,333 sequences

The dataset shows a notable class imbalance, with non-autophagy proteins forming the majority class.

# Dataset Structure

protein/
├── training dataset/
│   ├── autophagy protein/
│   │   └── autophagy.trn.fasta
│   └── non autophagy protein/
│       └── non-autophagy.trn.fasta
└── testing dataset/
├── autophagy protein/
│   └── autophagy.tst.fasta
└── non autophagy protein/
└── non-autophagy.tst.fasta

# Model Architecture

The project employs classical supervised machine learning models for binary classification. Instead of deep learning architectures, feature-based learning is used. The models applied in this project include:

* Support Vector Machine (SVM)
* Random Forest Classifier
* XGBoost Classifier

Each model is trained using extracted numerical features derived from protein sequences.

# Model Training

The training process begins with loading and preprocessing protein sequences, followed by feature extraction. The dataset is split into training and testing subsets. Machine learning models are trained using the training data, and hyperparameters are adjusted to achieve optimal performance. The models learn patterns that differentiate autophagy proteins from non-autophagy proteins.

# Training Results

The trained models demonstrate effective learning capability despite class imbalance. Performance is evaluated using multiple metrics to ensure robustness.

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Matthews Correlation Coefficient (MCC)**

Among the tested models, ensemble-based approaches such as Random Forest and XGBoost show strong performance.

# Data Preprocessing

Data preprocessing includes cleaning protein sequences, removing invalid characters, and standardizing input data. The sequences are converted into numerical feature vectors representing biochemical and physicochemical properties suitable for machine learning algorithms.

# Exploratory Data Analysis (EDA)

Exploratory analysis is performed to understand class distribution, sequence length variation, and dataset imbalance. This step provides insights that guide feature extraction and model selection.

# Class Weighting

Due to the imbalanced nature of the dataset, class weighting techniques are applied during training to reduce bias toward the majority class and improve minority class prediction.

# Model Evaluation

The trained models are evaluated on the test dataset using accuracy, precision, recall, F1-score, and MCC. Comparative analysis is performed to identify the best-performing model. Confusion matrices are also used to analyze classification behavior.

# Usage Example

The notebook includes step-by-step execution for data loading, preprocessing, training, and evaluation. Users can run the notebook sequentially to reproduce results and test model predictions on protein sequences.

# Future Improvements

* Incorporating deep learning approaches for sequence modeling
* Applying advanced feature extraction techniques
* Further optimization of hyperparameters
* Expanding the dataset with additional protein sequences

Feel free to explore, modify, and contribute to this project to further improve computational prediction of autophagy-related proteins.

