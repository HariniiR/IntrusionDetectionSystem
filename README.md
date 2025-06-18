

# Intrusion Detection System using a Neural Network (Built from Scratch)

This project implements an **Intrusion Detection System (IDS)** using a **neural network built entirely from scratch** in Python with **NumPy**. It is trained to classify **network traffic data** as either **normal** or **anomalous** based on features from the popular **KDD Cup 1999 dataset**.

---

## Features

* Neural Network built from scratch (no frameworks like TensorFlow or PyTorch)
* Uses **forward propagation**, **backpropagation**, and **gradient descent**
* One-hot encoding of categorical data (`protocol_type`, `service`, `flag`)
* Min-max normalization for numerical features
* Binary classification: `normal (0)` vs `anomaly (1)`
* Accuracy tracking during training

---

##  Dataset

The model is trained and tested on a cleaned version of the **KDD Cup 1999** dataset.

>  Format:

```
duration, protocol_type, service, flag, src_bytes, dst_bytes, ..., class
```

>  Labels:

* `normal`: Normal network activity → `0`
* `anomaly`: Any attack or suspicious behavior → `1`

---

## 📂 Project Structure

```
IntrusionDetectionSystem/
├── Trainedmodel.npz            # Saved model
├── datasets/                   # Folder for all CSV files
│   ├── training_data.csv       #Training data with lesser columns for a simpler model
│   ├── TestingData.csv
│   └── TrainingData.csv     
│
└── scripts/                    # All Python scripts
    ├── DataPreProcessing.py   # Preprocessing utilities
    ├── NeuralNetwork.py       # Model class with train/predict logic
    ├── Train.py               # Script to train the model
    ├── Predict.py             # Script to run predictions
    └── Predictor.py           # Script to load model & predict new input

```

---

## Neural Network Architecture

* **Input Layer**: Number of features after encoding
* **Hidden Layers**: 1 or more layers (configurable)
* **Activation Functions**: ReLU for hidden layers, Sigmoid for output
* **Loss Function**: Binary Cross Entropy
* **Optimizer**: Gradient Descent

---

## How to Run

### 1.  Install dependencies

```bash
pip install numpy pandas
```

### 2. Train the model

```bash
python IntrusionDetectionSystem/scripts/Train.py
```

### 3. Predict on new data

```bash
python src/IntrusionDetectionSystem/scripts/Predictor.py
```

---

##  Results

* Achieved **87.5% accuracy** on test data
* Model performance improves with more epochs (e.g., 1000+)

---

##  Key Learnings

* How neural networks work under the hood
* Preprocessing for categorical and numerical data
* Building ML models without high-level libraries
* Anomaly detection in cybersecurity

---

##  Tech Stack

* **Language**: Python
* **Libraries**: NumPy, Pandas

---

##  License

This project is for educational and demonstration purposes. You are free to use or extend it with proper attribution.


