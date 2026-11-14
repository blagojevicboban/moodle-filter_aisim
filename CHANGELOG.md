# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-11-14

### Added
- Data Preparation (EDA) simulation (`data-aisim="data-prep"`, also `data_prep` and `eda`): raw training data with
  missing values, a unit mistake (outliers) and two variables on very different scales. Statistics (minimum,
  maximum, mean, median, missing values), histograms by class, a pipeline of missing values (0, drop, mean,
  median), outliers (keep, remove) and scaling (none, min-max, z-score), and the space the k-NN model sees with the
  same scale on both axes. Accuracy and recall on clean test data. Datasets machines and patients. Book 2.6–2.8.
- Teacher guide 1.4 (`docs/`): Data Preparation module in all four editions.

## [0.12.0] - 2026-11-14

### Added
- Cross-Validation simulation (`data-aisim="cross-validation"`, also `cross_validation` and `k-fold`): the degree of a
  polynomial is the hyperparameter. The training data is split into k folds (2–10), each fold is held out once, and
  the mean and spread of the fold errors choose the degree (★). Charts of the data coloured by fold, the error of
  every fold and the training and cross-validation error by degree; 🎲 draws a new split and 🔒 unlocks a separate
  test set once. Datasets sine wave, sensor warm-up curve and very noisy measurements. Book chapters 3.9–3.11.
- Teacher guide 1.3 (`docs/`): Cross-Validation module in all four editions.

### Fixed
- `data-aisim="knn_tree"`, which the editor button writes, showed nothing; it is now an alias of `knn-tree`, and
  `data-model` chooses the initial model (k-NN or decision tree) for every k-NN alias.

## [0.11.0] - 2026-11-14

### Added
- Logistic Regression simulation (`data-aisim="logistic"`, also `logistic-regression`): two input variables,
  z = w1·x1 + w2·x2 + b, the sigmoid and a decision threshold θ. The weights are set by hand or learned with
  gradient descent on the log-loss; the probability map shows the straight decision boundary, the sigmoid chart
  places every point on the curve, and a tap on a point shows its calculation. Accuracy, precision, recall and the
  loss curve. Three datasets: machine failure, exam result (overlapping classes) and a ring that no straight line
  can separate. Book chapter 3.6.
- Teacher guide 1.2 (`docs/`): Logistic Regression module in all four editions.

### Fixed
- `data-readonly="1"`, set by the editor button's read-only option, now locks the controls of every simulation;
  the filter did not pass it to the simulations before.

## [0.10.1] - 2026-11-14

### Changed
- The layouts of the nine simulations are Mustache templates (`templates/`) rendered with `core/templates`
  instead of HTML strings in JavaScript; the shared slider is a partial. The rendered markup is unchanged
  (reviewer issue #3).
- The pixel and cell calculation boxes of Convolution and Q-Learning are built as DOM nodes, without `innerHTML`.
- CI also runs the Mustache linter.

### Fixed
- Full Moodle GPL boilerplate header in all AMD source files (reviewer issue #2).

## [0.10.0] - 2026-10-31

### Changed
- All simulation texts live in the language files (265 strings) and are loaded with `core/str`, so the simulations
  follow the user's language and can be translated in AMOS. The engines take them as `options.strings` and list the
  identifiers they use in `Engine.STRINGS`.
- Teacher guide 1.1 (`docs/`): sections for all nine simulations, a suggested lesson order and the meaning of the
  question targets; Slovak and Portuguese editions.

### Added
- Slovak and Portuguese translations; the Serbian Cyrillic translation is complete. They are kept in the repository;
  the release package contains English only (other languages go through AMOS).
- PHPUnit test that the engines, their string lists and every language file agree (keys and placeholders).

### Fixed
- Serbian words in the English interface (decision tree depth, regression and k-means labels, confusion matrix cells).

### Removed
- About 60 language strings no code used.

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
