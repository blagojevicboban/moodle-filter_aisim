# AI Simulations Viewer (Moodle Filter)

[![Moodle Plugin CI](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml/badge.svg)](https://github.com/blagojevicboban/moodle-filter_aisim/actions/workflows/ci.yml)
[![Moodle](https://img.shields.io/badge/Moodle-4.5%20to%205.3-orange.svg)](https://moodle.org)
[![PHP](https://img.shields.io/badge/PHP-8.2%2B-purple.svg)](https://php.net)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

**AI Simulations Viewer** (`filter_aisim`) is a Moodle text filter that renders **interactive, real-time AI and machine learning simulations** inside Moodle Pages, Lessons, Quizzes, or Books.

Designed specifically for vocational education (VET) and aligned with the **[AI for VET](https://aiforvet.eu/)** curriculum (*IO1: Fundamentals of AI*).

> Status: **alpha** (v0.1.0).

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📉 **Interactive Simulations** | 100% client-side (Canvas/SVG/JS) — no server load, zero latency, no API keys needed. |
| 🎛️ **Live Hyperparameters** | Students adjust Learning Rate ($\alpha$), initial weights ($w_0$), thresholds, and epochs with instant visual feedback. |
| 📊 **Real-time Metrics** | Cost function $J(w)$, gradient $\frac{dJ}{dw}$, convergence detection, and loss curves. |
| 🌍 **Multilingual** | English, Serbian Latin (`sr_lt`), and Serbian Cyrillic (`sr_cr`). |
| 🔒 **GDPR Compliant** | Implements Moodle's Null Privacy Provider — no personal data is collected or transmitted. |

---

## 📚 Simulations Catalog

1. **Gradient Descent Lab** (`data-aisim="gradient-descent"`):
   - Convex functions (single global minimum).
   - Non-convex functions (local minima and plateaus).
   - Real-time step trajectory, tangent slopes, and cost curves.

*(Upcoming: Confusion Matrix & ROC Curve, Polynomial Regression Overfitting, Decision Trees / k-NN Decision Boundaries, K-Means Clustering, CNN Kernel Visualizer)*

---

## 💻 Usage in Moodle

Simply place a container tag with the `data-aisim` attribute in any text area:

```html
<div data-aisim="gradient-descent"
     data-profile="convex"
     data-lr="0.15"
     data-w0="-1.0"></div>
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
