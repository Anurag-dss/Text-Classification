## 📌 Project Overview

- **Problem Type:** Binary Text Classification  
- **Dataset:** IMDB Movie Reviews  
- **Task:** Predict sentiment (Positive / Negative)  
- **Framework:** TensorFlow & Keras  
- **Platform:** Google Colab  

The model analyzes text data and classifies unseen movie reviews based on sentiment.

---

## 📂 Dataset Information

The project uses the **IMDB Reviews dataset**:

- **50,000 movie reviews**  
- **25,000 training samples**  
- **25,000 testing samples**  
- Reviews are **integer-encoded** for neural network input

📚 Official dataset reference:  
https://www.tensorflow.org/datasets/catalog/imdb_reviews  

---

## 🔄 Data Processing Pipeline

### 1️⃣ Data Loading
- Dataset loaded using **TensorFlow Datasets (TFDS)**
- Reviews are integer-encoded for neural network compatibility

### 2️⃣ Data Preparation
- Reviews are padded to ensure equal sequence length
- Labels are binary (0 = Negative, 1 = Positive)

### 3️⃣ Data Decoding (Human-readable)
- Integer sequences converted back to readable text
- Uses the **official Keras IMDB word index**

📚 Reference:  
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb  

---

## 🧠 Model Architecture

- **Embedding Layer:** Converts words into dense vector representations  
- **Dense Hidden Layer (ReLU):** Learns semantic patterns in the text  
- **Output Layer (Sigmoid):** Predicts probability of positive sentiment  

📚 Model design reference:  
https://www.tensorflow.org/tutorials/keras/text_classification  

---

## ⚙️ Model Training

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metric:** Accuracy  

The model was trained for multiple epochs and evaluated on unseen test data.

---

## 📊 Model Performance

- **Test Accuracy:** ~85%  
- **Loss:** ~0.40  

📚 Metric explanation:  
https://developers.google.com/machine-learning/crash-course/classification/accuracy  

---

## 📈 Visualizations

### Training & Validation Accuracy
<img width="716" height="537" alt="image" src="https://github.com/user-attachments/assets/8818ee76-a77e-458e-968c-105027b4181f" />


### Training & Validation Loss
<img width="717" height="537" alt="image" src="https://github.com/user-attachments/assets/ad3f0e3c-9765-42fb-b56a-687addf48dc5" />


### Review Length Distribution
<img width="784" height="526" alt="image" src="https://github.com/user-attachments/assets/0ea61c03-b464-42a2-928d-76d39a64e2d7" />


### Sample Predictions (Decoded Reviews)
REVIEW:
if not then it its officers was whether america's is this audience and <UNK> violence enjoy was paced genius <START> was started targeted basinger <START> richard a it mary <START> <START> live wanted <START> jamie began <UNUSED> that lester <UNUSED> that describe that too and <UNK> 60s made br romantic opportunity in believe making br right <UNK> truly carmen <UNUSED> when don't br him after br opportunity lot in believe so a laughing <START> turned of cards <START> of so stoic truly and scientist <START> it he she are a rent around as the front and in there's <UNUSED> also the <START> wakes and is a thinks language <UNK> keaton's <START> <UNUSED> <START> and is look seeing of camp flop the ...

Actual: Positive
Predicted: Positive
====================================================================================================
REVIEW:
the <START> paul hate and the <START> par <START> depend <UNK> fiasco it leaders <START> this close a burns to two close a rest the location <START> transformers as the bergman and manipulate that by <START> they you murphy slightly and movie away is this by <START> turned a numbing as many put <UNUSED> kid shut <START> <START> seems request watch and <START> acted murphy then to aspect and <UNK> being heston <START> of high beautifully <UNUSED> <START> i <UNK> nearly reflects of go it this scheme part to <START> most worth those said to doubts is consists <UNK> noise o was pulled <UNUSED> <START> chan the forward it psyche too remarkably and anchors poor <START> seems of apart good <START>  ...

Actual: Positive
Predicted: Positive
====================================================================================================
REVIEW:
usually for proud trust for narrative for killing <UNK> <START> film old joan for <UNUSED> old portrayed for away lab for <UNK> <START> to the original it events <START> gory <UNK> lake themselves now horrible sounds is br are the satisfying park was seasons scene that alcoholic <UNUSED> <UNK> drive clever with on best gone <START> the chasing so people what on had their <UNK> skills the ward does to young <UNK> my though it bit which <START> on bad makes a their in actions of <UNK> writers new with latest <START> larry <START> i on had rest his tells <UNK> put <START> it or his <START> a why around as <UNK> sight from the there <START> she to no make of his warm around in themselves with ma ...

Actual: Negative
Predicted: Negative
====================================================================================================
REVIEW:
mind tender <START> sound believe a unimaginative has <START> a in <START> <START> palace for with supposed <START> at <UNK> ferrell cyborg and the movie again film <UNK> comedy interesting <START> to <START> i in this run at striking <START> little his are the <START> making me whole voice they how <START> had also died final <START> as almost that <START> at stop dr so br him best some took a how man in of <UNK> might and for it back us a plane it love of outdated <UNUSED> <START> <UNUSED> it lying <START> chorus seems trio an bit have bit its was the left matrix seeing with without save that barely – it's <UNK> into real <UNK> movie are we such script notch from some <UNK> movies to <UNK> ...

Actual: Negative
Predicted: Negative
====================================================================================================
REVIEW:
that the we're <START> difference br son in movie more well is this so figure <UNUSED> new is which offers time <UNK> escape it obnoxious stand <START> opera br white watch do opera been atmosphere in including could seems br stop it's in for of all it isn't a think was into as no months including with one comedy he and up completely who about seen weak but <START> get son is more well br fact is presented out and effects <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD ...

Actual: Positive
Predicted: Positive

---

## 📊 Confusion Matrix

A **confusion matrix** shows how well the sentiment analysis model classifies reviews by comparing **actual labels** with **predicted labels**.

### What It Represents

| Term                | Meaning                                     |
| ------------------- | ------------------------------------------- |
| True Negative (TN)  | Negative review correctly classified        |
| False Positive (FP) | Negative review incorrectly classified as positive |
| False Negative (FN) | Positive review incorrectly classified as negative |
| True Positive (TP)  | Positive review correctly classified        |

**Balanced precision and recall** are important to ensure the model identifies both positive and negative reviews correctly.


<img width="727" height="534" alt="image" src="https://github.com/user-attachments/assets/6d24362e-0038-42b4-a633-1739f7817f48" />



### Interpretation

* **TP / TN:** Correct classifications
* **FP / FN:** Misclassifications
* **Accuracy ≈ 85%** → good overall performance
* **Precision & Recall** per class can be viewed using:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
```

---

## 🚀 Tools & Technologies Used

* Python
* TensorFlow
* TensorFlow Datasets
* Keras
* NumPy
* Matplotlib
* Seaborn
* Google Colab

---

## 🎓 What I Learned

* Loading and preprocessing text data
* Word embeddings and neural network concepts
* Building and training neural networks
* Evaluating classification models
* Visualizing results
* Interpreting predictions

This project provided a **strong foundation in NLP and deep learning**.

---

## 📌 Future Improvements

* Use **pretrained models from TensorFlow Hub**
* Add **LSTM / GRU layers**
* Implement **ROC Curve & Normalized Confusion Matrix**
* Try **transformer-based models (BERT)**

📚 Reference:
[https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub)

---

## 📖 References

1. TensorFlow Text Classification Tutorial
   [https://www.tensorflow.org/tutorials/keras/text_classification](https://www.tensorflow.org/tutorials/keras/text_classification)

2. IMDB Dataset Documentation
   [https://www.tensorflow.org/datasets/catalog/imdb_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)

3. Keras IMDB Word Index
   [https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)

4. Scikit-learn Confusion Matrix & Metrics
   [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 🙌 Acknowledgements

This project is inspired by official **TensorFlow tutorials**, created as part of my learning journey into **Natural Language Processing and AI**.
