# AI Simulations Viewer (Moodle Filter)

[![Moodle Plugin CI](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml/badge.svg)](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml)
[![Moodle](https://img.shields.io/badge/Moodle-4.5%20to%205.3-orange.svg)](https://moodle.org)
[![PHP](https://img.shields.io/badge/PHP-8.2%2B-purple.svg)](https://php.net)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

**AI Simulations Viewer** (`filter_aisim`) is a Moodle text filter that renders **interactive, real-time AI and machine learning simulations** inside Moodle Pages, Lessons, Quizzes, or Books.

Designed specifically for vocational education (VET) and aligned with the **[AI for VET](https://aiforvet.eu/)** curriculum (*IO1: Fundamentals of AI*).

> Status: **alpha** (v0.2.0).

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📉 **Interactive Simulations** | 100% client-side (Canvas/SVG/JS) — no server load, zero latency, no API keys needed. |
| 🎛️ **Live Hyperparameters & Thresholds** | Students adjust Learning Rate ($\alpha$), initial weights, decision thresholds ($\theta$), and observe metrics in real time. |
| 📊 **Dynamic Metrics & ROC Curves** | Confusion Matrix (TP, FP, TN, FN), Accuracy, Precision, Recall, F1-score, and ROC/AUC curves. |
| 🏭 **Real VET Scenarios** | Scenarios include Industrial Quality Control (Defect Detection), Medical Screening, and Spam Filtering. |
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
- Pedagogical guidance tailored to VET scenarios (impact of Type I vs Type II errors).

*(Upcoming: Polynomial Regression Overfitting, Decision Trees / k-NN Decision Boundaries, K-Means Clustering, CNN Kernel Visualizer)*

---

## 💻 Usage in Moodle

### Gradient Descent
```html
<div data-aisim="gradient-descent"
     data-profile="convex"
     data-lr="0.15"
     data-w0="-1.0"></div>
```

### Confusion Matrix
```html
<div data-aisim="confusion-matrix"
     data-scenario="quality"
     data-threshold="0.50"></div>
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

## 📄 License

Licensed under the [GNU General Public License v3 or later](LICENSE).
Copyright © 2026 Boban Blagojević.
