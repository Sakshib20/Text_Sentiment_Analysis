# Text Sentiment Analysis using RNN/LSTM

An industrial-grade Natural Language Processing (NLP) pipeline implementing a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers in TensorFlow/Keras. The system classifies unstructured text datasets (e.g., movie reviews, social media posts) into Positive, Negative, or Neutral sentiments by modeling sequential dependencies and semantic context within a unified codebase.

# 📌 Project Overview

Unlike bag-of-words or naive machine learning models that ignore word order, this project implements sequential deep learning. The application tokenizes raw text, maps it to a dense vector embedding space, and processes it through an LSTM architecture to capture long-term context and dependencies.

This implementation features an end-to-end, zero-dependency data-to-inference pipeline contained entirely within a single, highly modular execution script.

# 🛠 Tech Stack & Dependencies

Language: Python 3.x

Deep Learning Framework: TensorFlow 2.x, Keras (Sequential API)

Natural Language Processing: NLTK (Tokenization, Lexical Filtering, Stopword Removal)

Scientific Computing & Data Manipulation: NumPy, Pandas, Scikit-learn

Visualization: Matplotlib

# ⚙️ Core Architecture & Pipeline Logic

The application processes text and executes training through a sequential, modular pipeline embedded in Text_Sentiment_Analysis.py:

[ Raw Unstructured Text ]
         │
         ▼  preprocess_text() -> NLTK Stopword Filtering
[ Normalized Tokens ]
         │
         ▼  pad_sequences() -> Shape: [Batch Size, Max Sequence Length]
[ Padded Sequences ]
         │
         ▼  Keras Embedding Layer -> Vector Dimension (d)
[ Continuous Embeddings ]
         │
         ▼  LSTM Layer(s) with Spatial & Recurrent Dropout
[ Contextual Sequence Vectors ]
         │
         ▼  Dense Classifier Head -> Softmax Activation
[ Sentiment Class Probabilities ]


# 1. Preprocessing & Vectorization

Text Normalization: Lowercases incoming strings, strips computational noise (HTML tags, URLs, special characters), and tokenizes sentences.

NLTK Filtering: Eliminates low-information stopwords (e.g., "is", "the", "at") to compress vocabulary size and reduce computation time.

Sequence Padding: Pads or truncates sequences to a uniform sequence length $T_x$ using post padding to enable efficient batch matrix multiplication.

Word Embeddings: Maps sparse token indices to continuous dense vectors of dimension $d$ to maintain semantic similarities.

2. Recurrent Network Topology

LSTM Gating: Employs memory cells with input, forget, and output gates to combat the vanishing gradient problem, allowing the system to preserve long-range text context.

Regularization: Integrates a combination of standard Dropout and Recurrent Dropout ($0.2$ to $0.5$) to decouple node dependencies and prevent overfitting on noisy user data.

Classifier Head: Maps the temporal sequence outputs to a dense classifier head with a Softmax activation to generate probability distributions across classes.

3. Execution Pipeline & Flow

The script functions through three discrete stages:

Data Ingestion & Splits: Loads, cleans, tokenizes, and splits the data into training, validation, and test arrays.

Model Training: Compiles the network with the Adam Optimizer and Cross-Entropy loss, executing mini-batch gradient descent.

Inference & Metrics: Generates training curves and processes custom text input strings for validation.

# 📂 Repository Structure

├── Text_Sentiment_Analysis.py  # Unified script handling preprocessing, training, and evaluation
├── README.md                   # Project documentation and system architecture overview
└── requirements.txt            # Production dependencies (TensorFlow, NLTK, NumPy, Scikit-learn)


# 🚀 Execution & Usage Guide

1. Setup Environment

Ensure your Python environment is configured, then clone and install the dependencies:

# Clone the repository
git clone [https://github.com/Sakshib20/Text_Sentiment_Analysis.git](https://github.com/Sakshib20/Text_Sentiment_Analysis.git)
cd Text_Sentiment_Analysis

# Install version-locked libraries
pip install -r requirements.txt


2. Run the Complete Pipeline

Executing the main script triggers the entire workflow—from downloading NLTK dependencies and preprocessing raw datasets to compiling, training, and outputting performance graphs:

python Text_Sentiment_Analysis.py


# 📊 Evaluation & Diagnostics

The script generates training history curves to evaluate model performance:

Overfitting Verification: If the training loss curves continue to fall while the validation loss plateaus or climbs, it highlights High Variance (Overfitting). Fix this by adjusting the dropout rate or decreasing the embedding vector dimension $d$ inside the hyperparameters section.

Precision, Recall, & F1-Score: The evaluation suite utilizes a Scikit-learn classification report to calculate performance metrics across all classes, protecting the system from minority-class performance collapse in imbalanced data scenarios.
