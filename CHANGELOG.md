# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-10-30

### Added
- PHPUnit tests for the filter.

### Changed
- Maturity: beta.

### Removed
- Development files that only worked on the author's machine: `demo/create_moodle_demo.php` and the `test.html`
  playground (use https://sim.aiforvet.eu instead).

## [0.9.0] - 2026-10-30

### Added
- Data Split & Bias simulation (`data-aisim="data-bias"`): a k-NN model checks parts from two factories with a
  different defect rule. Sample size, share of factory B, test split and k; accuracy on the training set, on the
  held-out test set and on a representative "real world" set per factory. Shows how a biased sample hides a model
  that fails for the under-represented group, and why training accuracy (k = 1: 100%) means nothing.

## [0.8.0] - 2026-10-30

### Added
- Reinforcement Learning simulation (`data-aisim="q-learning"`): a Q-learning agent in a grid world (maze, cliff,
  near-or-far treasure) with learning rate α, discount factor γ and exploration ε. Shows state values, the best move
  per cell, the learned path from the start, the reward per episode and the Q-values of a tapped cell. Seeded
  episodes: the same settings and `data-episodes` always give the same trained agent.

## [0.7.0] - 2026-10-30

### Added
- Convolution & Image Filters simulation (`data-aisim="convolution"`): a 3×3 kernel over a 28×28 grayscale image
  (shapes, stripes, smiley, shapes with noise). Ready-made filters (blur, Gaussian blur, sharpen, Sobel, Laplacian,
  emboss) or typed weights, divisor and absolute value. Tapping a pixel shows its calculation. With a target filter
  (`data-target`) the output is compared with the target image (R²); `data-presets="false"` hides the ready-made filters.

## [0.6.0] - 2026-10-30

### Added
- Perceptron & Neural Network simulation (`data-aisim="neural-network"`): datasets linear / XOR / circle in a circle,
  0–2 hidden layers with 1–8 neurons, sigmoid / tanh / ReLU, learning rate, live decision boundary, network drawing
  (line width = weight) and loss curve. Seeded data and weights: the same settings and `data-epochs` give the same
  trained network.

### Changed
- `.gitattributes`: LF line endings, as in the other aisim plugins.

### Removed
- The demo course backup (`demo/aiforvet_simulations_demo_course.mbz`) and the teacher guide section on restoring it.

## [0.5.0] - 2026-10-28

### Added
- `KMeansSim`: Interactive unsupervised K-Means clustering simulation engine.
- Step-by-step execution: Alternating assignment phase (lines to centroids) and update phase (centroids moving to cluster mean).
- Live Inertia (WCSS) calculation and Elbow Method plot ($K=1\dots 6$).
- Random centroid re-seeding to explore local minima sensitivity.
- Interactive custom data point insertion on canvas click.
- 5th demo lesson: `demo/05_kmeans_klasterovanje.html`.
- 5th activity in Moodle 4.5 demo course (ID=77).
- Full compilation of AMD modules via Moodle Grunt with 0 ESLint errors.

## [0.4.0] - 2026-10-28

### Added
- `KnnTreeSim`: Interactive 2D decision boundary visualizer comparing k-Nearest Neighbors (k-NN) and Decision Trees.
- Fast 2D decision space rasterizer rendering smooth vs orthogonal (axis-aligned) decision boundaries.
- Interactive mouse hover in k-NN mode highlighting the $k$ nearest neighbors with dashed distance vectors and enclosing circle.
- 4th demo lesson: `demo/04_granice_odlucivanja.html`.

## [0.3.0] - 2026-10-28

### Added
- `RegressionSim`: Interactive polynomial curve fitting, degree tuning ($d=1\dots 9$), and Ridge regularization ($\lambda$).
- Automatic Least Squares solver with Ridge penalty: $(X^T X + \lambda I) w = X^T y$ via Gaussian elimination.
- Live Model Complexity vs Error (U-Curve) plotting Train MSE vs Validation MSE.
- 3rd demo lesson: `demo/03_polinomska_regresija.html`.

## [0.2.0] - 2026-10-28

### Added
- `ConfusionMatrixSim`: Interactive 2x2 confusion matrix and classification threshold tuner.
- Probability density curves with draggable threshold handle $\theta$.
- Real-time performance metrics (Accuracy, Precision, Recall, F1-score) with animated progress indicators.
- Live ROC curve ($FPR$ vs $TPR$) with AUC computation and operating point tracking.
- 2nd demo lesson: `demo/02_matrica_konfuzije.html`.

## [0.1.0] - 2026-10-28

### Added
- Initial project structure for `filter_aisim` supporting Moodle 4.5 to 5.3.
- `GradientDescentSim`: Interactive 2D gradient descent lab supporting convex and non-convex cost functions.
- Trilingual localization (`en`, `sr_lt`, `sr_cr`).
- Moodle Plugin CI workflow for GitHub Actions.
- Standalone test playground (`test.html`) and 1st demo lesson (`demo/01_gradijentni_spust.html`).
