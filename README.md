

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
├── DataPreProcessing.py     # Handles encoding and normalization
├── NeuralNetwork.py         # Neural net logic: init, train, predict
├── Train.py                 # Training script
├── Predict.py               # Prediction logic on new samples
├── Predictor.py             # Run predictions from saved model
├── training_data.csv        # Training data
├── testing_data.csv         # Test samples
└── README.md                # Project documentation
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
python src/IntrusionDetectionSystem/Train.py
```

### 3. Predict on new data

```bash
python src/IntrusionDetectionSystem/Predictor.py
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


