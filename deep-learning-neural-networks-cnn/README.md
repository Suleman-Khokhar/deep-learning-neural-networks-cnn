# Image Classification with Neural Networks
### A Comparative Study of Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) on MNIST and Fashion-MNIST

<p align="center">
  <strong>Exploring neural network architectures, data augmentation, knowledge distillation, and model performance.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-orange" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Datasets-MNIST%20%7C%20Fashion--MNIST-green" alt="Datasets">
  <img src="https://img.shields.io/badge/Project-Research%20%26%20Learning-purple" alt="Project">
</p>

---

## Overview

How do different neural network architectures perform when classifying handwritten digits and clothing images?

This project explores that question through a comparative study of **Artificial Neural Networks (ANNs)** and **Convolutional Neural Networks (CNNs)** using two well-known image classification datasets: MNIST and Fashion-MNIST.

The experiments investigate how model architecture and training strategies influence classification performance. In particular, the project explores data augmentation, knowledge distillation, and a combined training strategy to understand whether these techniques can improve the performance of fully connected neural networks.

Rather than focusing only on accuracy, this project examines the strengths and limitations of different approaches through quantitative evaluation and error analysis.

## Project Objectives

- Implement and evaluate Artificial Neural Network architectures for image classification.
- Establish a Convolutional Neural Network as a comparison baseline.
- Investigate the impact of data augmentation on ANN performance.
- Explore knowledge distillation using a trained teacher model.
- Evaluate a combined training strategy.
- Compare model performance across two different image classification datasets.
- Analyse classification errors and understand the limitations of each approach.

## Datasets

The experiments use two datasets with different classification characteristics.

### 1. MNIST — Handwritten Digit Recognition

MNIST contains grayscale images of handwritten digits from 0 to 9.

| Property | Description |
|---|---|
| Image dimensions | 28 × 28 pixels |
| Colour channels | 1 (grayscale) |
| Number of classes | 10 |
| Classes | Digits 0–9 |
| Task | Handwritten digit classification |

Dataset: [MNIST — TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)

### 2. Fashion-MNIST — Clothing Classification

Fashion-MNIST contains grayscale images representing ten categories of clothing and accessories.

| Property | Description |
|---|---|
| Image dimensions | 28 × 28 pixels |
| Colour channels | 1 (grayscale) |
| Number of classes | 10 |
| Task | Clothing image classification |

**Classes:**

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Dataset: [Fashion-MNIST — Zalando Research](https://github.com/zalandoresearch/fashion-mnist)

### Why use both datasets?

MNIST provides relatively simple visual patterns, while Fashion-MNIST contains more visually similar categories.

Comparing both datasets helps demonstrate how the complexity of image features can influence the performance of different neural network architectures.

---

## Model Architectures and Experimental Configurations

The project compares a CNN baseline against four ANN configurations.

| Model | Configuration | Purpose |
|---|---|---|
| CNN | Convolutional Neural Network | Reference model for comparison |
| A0 | Baseline ANN | Establish baseline ANN performance |
| A1 | ANN with data augmentation | Investigate the effect of augmented training data |
| A2 | ANN with knowledge distillation | Investigate teacher-guided learning |
| A3 | Combined training strategy | Evaluate the combination of augmentation and distillation |

### CNN — Baseline Model

The CNN serves as the reference architecture.

Convolutional layers learn spatial features from images, while pooling layers reduce spatial dimensions and retain important information. The extracted features are subsequently used for classification.

CNNs are particularly suitable for image classification because they preserve and learn spatial relationships between neighbouring pixels.

### A0 — Baseline ANN

The baseline ANN uses fully connected layers to classify flattened image inputs.

Unlike CNNs, fully connected networks do not explicitly preserve the two-dimensional spatial structure of an image.

This configuration establishes a reference point for evaluating the improvements introduced by subsequent training strategies.

### A1 — ANN with Data Augmentation

This experiment investigates whether training with modified versions of the original images improves generalization.

Data augmentation introduces variations into training images while attempting to preserve their original labels.

The notebook explores augmentation as a way to expose the network to a broader range of input patterns.

### A2 — ANN with Knowledge Distillation

Knowledge distillation uses a trained teacher model to guide the learning process of a student model.

Instead of learning exclusively from the original class labels, the student can also learn from the teacher's predicted class probabilities.

This provides additional information about relationships between possible classes.

### A3 — Combined Training Strategy

The combined configuration investigates the use of data augmentation and knowledge distillation together.

The objective is to evaluate whether combining these techniques provides additional benefits compared with using either technique independently.

---

## Experimental Results

The following table presents the classification accuracy obtained in the experiments.

**Accuracy comparison**

| Dataset | CNN Baseline | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|---:|
| MNIST | **98.95%** | 97.95% | 98.18% | 98.21% | 98.59% |
| Fashion-MNIST | **90.64%** | 87.79% | 83.03% | 88.83% | 84.43% |

*Results are reported from the project experiments. Verify them against the final executed notebook before publication.*

### Accuracy Gap Relative to CNN

The following table shows how many percentage points each ANN configuration falls below the CNN baseline.

| Dataset | A0 Gap | A1 Gap | A2 Gap | A3 Gap |
|---|---:|---:|---:|---:|
| MNIST | 1.00 pp | 0.77 pp | 0.74 pp | 0.36 pp |
| Fashion-MNIST | 2.85 pp | 7.61 pp | 1.81 pp | 6.21 pp |

*pp = percentage points.*

### Key Observations

**MNIST**

- The CNN achieves 98.95% accuracy.
- A3 achieves the strongest performance among the ANN configurations at 98.59%.
- A3 reduces the accuracy gap to 0.36 percentage points.
- All four ANN configurations perform relatively close to the CNN baseline.

**Fashion-MNIST**

- The CNN achieves 90.64% accuracy.
- A2 achieves the strongest ANN performance at 88.83%.
- A2 has an accuracy gap of 1.81 percentage points.
- A1 and A3 perform less effectively than the baseline ANN in these experiments.

### Overall Interpretation

The results demonstrate that the effectiveness of a training strategy depends on the dataset.

The combined strategy produces the strongest ANN result on MNIST, whereas knowledge distillation produces the strongest ANN result on Fashion-MNIST.

This indicates that combining multiple training techniques does not automatically guarantee better classification performance.

---

## Evaluation and Error Analysis

Accuracy alone does not provide a complete understanding of a classification model.

The project also considers additional evaluation techniques to examine model performance and classification errors.

### Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified samples.
- **Precision:** Measures how many predictions for a particular class are correct.
- **Recall:** Measures how many actual samples of a class are correctly identified.
- **F1-score:** Combines precision and recall into a single metric.
- **Macro F1-score:** Calculates the F1-score independently for each class and averages the results equally.

### Confusion Matrix

A confusion matrix provides a detailed view of the model's predictions.

It helps identify:

- Correctly classified samples.
- Frequently confused classes.
- Classes that are difficult to distinguish.
- Differences in classification behaviour between datasets.

This is particularly useful for Fashion-MNIST, where some clothing categories have similar visual characteristics.

---

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Neural network implementation |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Jupyter Notebook | Experiment development and documentation |
| Scikit-learn | Evaluation metrics and analysis |

*Confirm the exact libraries and versions against the notebook environment.*

---

## Project Structure

```text
deep-learning-neural-networks-cnn/
│
├── image_classification_ann_vs_cnn.ipynb
│
├── README.md
│
├── requirements.txt
│
├── .gitignore
│
└── LICENSE
```

The Jupyter Notebook contains the implementation, experiments, model evaluation, visualisations, and discussion of the results.

---

## Getting Started

Follow these instructions to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Suleman-Khokhar/deep-learning-neural-networks-cnn.git
```

### 2. Navigate to the Project Directory

```bash
cd deep-learning-neural-networks-cnn
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
image_classification_ann_vs_cnn.ipynb
```

Run the notebook cells sequentially to reproduce the experiments.

---

## Reproducibility

For meaningful comparisons, experiments should be conducted under documented and consistent conditions.

The notebook should specify:

- Random seed settings.
- Dataset preprocessing.
- Model architecture and initialization.
- Optimizer and learning rate.
- Batch size and number of epochs.
- Data augmentation parameters.
- Knowledge distillation configuration.
- Evaluation procedure.

Results may vary depending on the software environment, hardware, random initialization, and training configuration.

---

## Limitations

This project has several limitations:

1. The experiments are restricted to relatively small grayscale image datasets.
2. The results do not establish that one architecture is universally better than another.
3. Performance depends on the selected hyperparameters and training configuration.
4. Knowledge distillation and augmentation may not improve performance in every setting.
5. The experiments do not establish how these approaches perform on large-scale, real-world image datasets.

These limitations provide opportunities for further experimentation.

---

## Future Improvements

Potential directions for extending this project include:

- Experimenting with deeper CNN architectures.
- Testing different ANN layer configurations.
- Comparing multiple data augmentation techniques.
- Investigating different teacher models and distillation temperatures.
- Performing systematic hyperparameter tuning.
- Comparing model training time and computational requirements.
- Evaluating models on additional image datasets.
- Exploring model interpretability techniques such as Grad-CAM for convolutional models.

---

## Learning Outcomes

Through this project, I explored several important concepts in deep learning:

- Differences between fully connected and convolutional architectures.
- The importance of spatial feature extraction in image classification.
- The role of data augmentation in model generalization.
- Teacher–student learning through knowledge distillation.
- The effect of training strategies on classification performance.
- The importance of evaluation metrics beyond accuracy.
- The challenges of comparing machine learning models fairly.

---

## Author

**Suleman Khokhar**

BS Computer Science Student  
Forman Christian College University (FCCU)

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Software Engineering.

<p align="left">
  <a href="https://github.com/Suleman-Khokhar">
    <img src="https://img.shields.io/badge/GitHub-Suleman--Khokhar-181717?style=for-the-badge&logo=github" alt="GitHub Profile">
  </a>
</p>

---

## Acknowledgements

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

---

<p align="center">
  <strong>Exploring the fundamentals of deep learning, one experiment at a time.</strong>
</p>
