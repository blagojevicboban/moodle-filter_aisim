# AI Simulations Viewer (Moodle Filter)

[![Moodle Plugin CI](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml/badge.svg)](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml)
[![Moodle](https://img.shields.io/badge/Moodle-4.5%20to%205.3-orange.svg)](https://moodle.org)
[![PHP](https://img.shields.io/badge/PHP-8.2%2B-purple.svg)](https://php.net)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

**AI Simulations Viewer** (`filter_aisim`) is a Moodle text filter that renders **interactive, real-time AI and machine learning simulations** inside Moodle Pages, Lessons, Quizzes, or Books.

Designed specifically for vocational education (VET) and aligned with the **[AI for VET](https://aiforvet.eu/)** curriculum (*IO1: Fundamentals of AI*).

> Status: **alpha** (v0.5.0).

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
| 🌍 **Multilingual** | English, Serbian Latin (`sr_lt`), and Serbian Cyrillic (`sr_cr`). |
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
- 📦 **[Turnkey Course Backup (.mbz)](https://github.com/blagojevicboban/moodle-qtype_aisim/releases/download/v0.5.2/aiforvet_simulations_demo_course.mbz)**

---

## 📄 License

Licensed under the [GNU General Public License v3 or later](LICENSE).
Copyright © 2026 Boban Blagojević.
