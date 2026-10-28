# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-28

### Added
- `RegressionSim`: Interactive polynomial curve fitting, degree tuning ($d=1\dots 9$), and Ridge regularization ($\lambda$).
- Automatic Least Squares solver with Ridge penalty: $(X^T X + \lambda I) w = X^T y$ via Gaussian elimination.
- Live Model Complexity vs Error (U-Curve) plotting Train MSE vs Validation MSE.
- Visual Underfitting (High Bias) vs Overfitting (High Variance) diagnostic indicators.
- Interactive custom data point insertion on canvas click.
- 3 built-in datasets: Sine wave with noise, Quadratic trend, and Engine RPM vs Fuel Consumption.
- Trilingual localization (`en`, `sr_lt`, `sr_cr`).
- 3rd tab in `test.html` playground.

## [0.2.0] - 2026-10-28

### Added
- `ConfusionMatrixSim`: Interactive 2x2 confusion matrix and classification threshold tuner.
- Probability density curves with draggable threshold handle $\theta$.
- Real-time performance metrics (Accuracy, Precision, Recall, F1-score) with animated progress indicators.
- Live ROC curve ($FPR$ vs $TPR$) with AUC computation and operating point tracking.
- Contextual VET scenarios: Industrial Quality Control, Medical Diagnostics, and Spam Filtering.

## [0.1.0] - 2026-10-28

### Added
- Initial project structure for `filter_aisim` supporting Moodle 4.5 to 5.3.
- `GradientDescentSim`: Interactive 2D gradient descent lab supporting convex and non-convex cost functions.
- Real-time step trajectory, tangent slope rendering, and live loss curve chart.
- Trilingual localization (`en`, `sr_lt`, `sr_cr`).
- Moodle Plugin CI workflow for GitHub Actions.
- Standalone test playground (`test.html`).
