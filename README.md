# AI Simulations Viewer (Moodle Filter)

[![Moodle Plugin CI](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml/badge.svg)](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml)
[![Moodle](https://img.shields.io/badge/Moodle-4.5%20to%205.3-orange.svg)](https://moodle.org)
[![PHP](https://img.shields.io/badge/PHP-8.2%2B-purple.svg)](https://php.net)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

**AI Simulations Viewer** (`filter_aisim`) is a Moodle text filter that renders **interactive, real-time AI and machine learning simulations** inside Moodle Pages, Lessons, Quizzes, or Books.

Designed specifically for vocational education (VET) and aligned with the **[AI for VET](https://aiforvet.eu/)** curriculum (*IO1: Fundamentals of AI*).

> Status: **stable** (v1.0.0).

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📉 **Interactive Simulations** | 100% client-side (Canvas/SVG/JS) — no server load, zero latency, no API keys needed. |
| 🎛️ **Live Hyperparameters & Thresholds** | Students adjust Learning Rate ($\alpha$), weights, classification thresholds ($\theta$), polynomial degrees, and $k$ clusters. |
| 📊 **Dynamic Metrics & ROC Curves** | Confusion Matrix (TP, FP, TN, FN), Accuracy, Precision, Recall, F1-score, and ROC/AUC curves. |
| 🌲 **Decision Boundaries** | Real-time comparison between k-NN (distance-based smooth/Voronoi boundaries) and Decision Trees (orthogonal splits). |
| 🫧 **Unsupervised Clustering** | Step-by-step K-Means algorithm (alternating assignment & centroid update) and live Elbow plot (WCSS). |
| 🏭 **Real VET Scenarios** | Scenarios include Industrial Quality Control (Defect Detection), Medical Screening, Fuel Consumption, and Sensor Regimes. |
| 🌍 **Multilingual** | Every simulation text comes from the Moodle language files, so the simulations follow the user's language. The package ships English; Serbian Latin (`sr_lt`), Serbian Cyrillic (`sr_cr`), Slovak (`sk`) and Portuguese (`pt`) translations are in this repository and go to Moodle sites through AMOS. |
| 🔒 **GDPR Compliant** | Implements Moodle's Null Privacy Provider — no personal data is collected or transmitted. |

---

## 📚 Simulations Catalog

### 1. Gradient Descent Lab (`data-aisim="gradient-descent"`)
- Convex quadratic curves (single global minimum).
- Non-convex curves with local minima and plateaus.
- Real-time step trajectory, tangent slopes, and cost history curve.

### 2. Confusion Matrix & Threshold Tuner (`data-aisim="confusion-matrix"`)
- Probability distributions $P(y=1|x)$ for positive and negative classes with draggable threshold $\theta$.
- Dynamic 2x2 confusion matrix (TP, FP, TN, FN).
- Real-time metric bars: Accuracy, Precision, Recall, and F1-score.
- Live ROC curve with animated operating point and AUC calculation.

### 3. Polynomial Regression & Overfitting (`data-aisim="regression"`)
- Degree tuning from $d=1$ (Underfitting / High Bias) up to $d=9$ (Overfitting / High Variance).
- L2 Ridge regularization ($\lambda$) slider.
- Live U-Curve comparing Train MSE vs Validation MSE.
- Interactive custom data point insertion on click.

### 4. Decision Boundaries: k-NN vs Decision Tree (`data-aisim="knn-tree"`)
- Smooth vs axis-aligned orthogonal 2D classification boundaries.
- Slider for $k$ neighbors ($1\dots 15$) and max tree depth ($1\dots 5$).
- Mouse hover distance vectors connecting query points to $k$ nearest neighbors.

### 5. K-Means Clustering Lab (`data-aisim="kmeans"`)
- Step-by-step alternating phases: 1. Assignment (points to nearest centroid); 2. Centroid update (moving to cluster mean).
- Live Within-Cluster Sum of Squares (Inertia / WCSS).
- Dynamic Elbow plot ($K=1\dots 6$) with current operating point.
- Centroid re-seeding to explore local minima sensitivity.

### 6. Perceptron & Neural Network (`data-aisim="neural-network"`)
- A perceptron (no hidden layer) separates only with a straight line: 100% on linear data, 50% on XOR.
- 1–2 hidden layers with 1–8 neurons and sigmoid / tanh / ReLU bend the boundary around XOR and a circle in a circle.
- Live decision boundary, network drawing (line width = weight strength) and loss curve.
- Seeded data and weights: the same settings and `data-epochs` always give the same trained network.

### 7. Convolution & Image Filters (`data-aisim="convolution"`)
- A 3×3 kernel slides over a small grayscale image: blur, Gaussian blur, sharpen, Sobel edges, Laplacian, emboss.
- Students type their own weights, divisor and absolute value; tapping a pixel shows the full calculation.
- Optional target filter (`data-target`): the output is compared with the target image (R²), `data-presets="false"`
  hides the ready-made filters so the kernel has to be built by hand.

### 8. Reinforcement Learning – Q-learning (`data-aisim="q-learning"`)
- An agent learns from rewards alone in a grid world: maze, cliff (the shortest path runs along the edge) and
  near-or-far (a small treasure close by, a big goal further away).
- Learning rate α, discount factor γ and exploration ε; cell colours show state values, arrows the best move,
  an orange line the learned path, a chart the reward per episode.
- Seeded episodes: the same settings and `data-episodes` always give the same trained agent.

### 9. Data Split & Bias (`data-aisim="data-bias"`)
- A k-NN model checks parts from two factories, A and B, where the defect rule differs.
- Sample size, share of factory B in the sample, held-out test share and k.
- Accuracy on training, on the test set and on a representative "real world" set per factory: a sample with almost
  no factory B looks excellent on its own test set and still fails for factory B.

### 10. Logistic Regression (`data-aisim="logistic"`)
- Two input variables, z = w1·x1 + w2·x2 + b, the sigmoid σ(z) and a decision threshold θ.
- Weights and bias by hand or learned with gradient descent on the log-loss; the probability map shows the straight
  decision boundary, the sigmoid chart places every point on the curve, a tap on a point shows its calculation.
- Accuracy, precision, recall and the loss curve; datasets machine failure, exam result and a ring that no straight
  line can separate.

### 11. Cross-Validation (`data-aisim="cross-validation"`)
- The degree of a polynomial (1–9) is the hyperparameter; the training data is split into k folds (2–10).
- Each fold is held out once: bars of the fold errors, their mean and spread, and the training and cross-validation
  error by degree (U-curve, best degree ★).
- 🎲 draws a new split (`data-seed`), 🔒 unlocks a separate test set once (`data-tested="1"`).

### 12. Data Preparation & EDA (`data-aisim="data-prep"`)
- Raw training data with missing values, a unit mistake and two variables on very different scales.
- Statistics and histograms, then a pipeline: missing values (0, drop, mean, median), outliers (keep, remove),
  scaling (none, min-max, z-score); the space the k-NN model sees, with the same scale on both axes.
- Accuracy and recall on clean test data; datasets machines and patients.

### 13. Recurrent Neural Network (`data-aisim="rnn"`)
- One recurrent neuron: h_t = tanh(w·h_(t-1) + u·x_t), y_t = v·h_t; sliders for w (memory), u and v, or training
  with gradient descent (+10 epochs).
- The sequence, the network unrolled over the last five steps and how strongly each past input still influences the
  last output (how many inputs the network remembers).
- Datasets: a noisy thermometer and a memory task with a sequence length L (`data-length`, 3–30) that shows
  forgetting and the vanishing gradient.

---

## 💻 Usage in Moodle

```html
<!-- Gradient Descent -->
<div data-aisim="gradient-descent" data-profile="convex" data-lr="0.15"></div>

<!-- Confusion Matrix -->
<div data-aisim="confusion-matrix" data-scenario="quality" data-threshold="0.50"></div>

<!-- Regression Overfitting -->
<div data-aisim="regression" data-dataset="sine" data-degree="3"></div>

<!-- Decision Boundaries -->
<div data-aisim="knn-tree" data-dataset="moons" data-k="3" data-depth="3"></div>

<!-- K-Means Clustering -->
<div data-aisim="kmeans" data-dataset="blobs3" data-k="3"></div>

<!-- Perceptron & Neural Network (optionally pre-trained with data-epochs) -->
<div data-aisim="neural-network" data-dataset="xor" data-layers="1" data-neurons="4" data-activation="tanh" data-lr="0.3"></div>

<!-- Convolution: a ready-made filter, or data-kernel="-1,0,1,-2,0,2,-1,0,1" data-divisor="1" data-abs="true";
     with a target to match and without the ready-made filters -->
<div data-aisim="convolution" data-image="shapes" data-preset="identity" data-target="sobely" data-presets="false"></div>

<!-- Q-learning (optionally pre-trained with data-episodes) -->
<div data-aisim="q-learning" data-map="cliff" data-alpha="0.5" data-gamma="0.9" data-epsilon="0.2"></div>

<!-- Data Split & Bias -->
<div data-aisim="data-bias" data-population="shifted" data-n="200" data-share="5" data-test="25" data-k="5"></div>

<!-- Logistic Regression (add data-readonly="1" to any simulation to lock its controls) -->
<div data-aisim="logistic" data-dataset="machine" data-w1="0" data-w2="0" data-b="0" data-threshold="0.5" data-lr="0.5"></div>

<!-- Cross-Validation -->
<div data-aisim="cross-validation" data-dataset="sine" data-k="5" data-degree="1" data-seed="1"></div>

<!-- Data Preparation & EDA -->
<div data-aisim="data-prep" data-dataset="machine" data-missing="zero" data-outliers="keep" data-scaling="none" data-k="5"></div>

<!-- Recurrent Neural Network (data-dataset="temperature" or "memory") -->
<div data-aisim="rnn" data-dataset="memory" data-length="10" data-w="0.5" data-u="1" data-v="1"></div>
```

---

## ⚙️ Requirements

- **Moodle**: 4.5 LTS or newer (tested on 4.5, 5.0, 5.1, 5.2, 5.3)
- **PHP**: 8.2+
- Modern Web Browser with HTML5 Canvas support.

---

## 📦 Installation

### Via Git
```bash
cd /path/to/moodle/public/filter   # (or filter/ in Moodle < 5.1)
git clone https://github.com/blagojevicboban/moodle-filter_aisim.git aisim
php admin/cli/upgrade.php
```

---

## 📚 Documentation & Resources

- 🇷🇸 **[Pedagoško-tehnički priručnik za nastavnike](docs/PRIRUCNIK_ZA_NASTAVNIKE.md)**
- 🇬🇧 **[Pedagogical and Technical Teacher Guide](docs/TEACHER_GUIDE.md)**
- 🇸🇰 **[Pedagogicko-technická príručka pre učiteľov](docs/PRIRUCKA_PRE_UCITELOV.md)**
- 🇵🇹 **[Guia pedagógico e técnico para professores](docs/GUIA_DO_PROFESSOR.md)**
- 🌐 Web version in all four languages: [sim.aiforvet.eu/guide.html](https://sim.aiforvet.eu/guide.html)

---

## 📄 License

Licensed under the [GNU General Public License v3 or later](LICENSE).
Copyright © 2026 Boban Blagojević.
