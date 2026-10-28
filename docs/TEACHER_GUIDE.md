# AI for VET – Pedagogical and Technical Guide for Teachers
## Integrating Interactive AI and Machine Learning Simulations into Moodle

*Project: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Programme: **Erasmus+***  
*Official Portal:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Authors and Technical Development Team of AI for VET*  
*Guide Version: 1.0 (October 2026)*  
*Compatibility: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Table of Contents
1. [Introduction and Pedagogical Framework](#1-introduction-and-pedagogical-framework)
   - 1.1. The Challenge: Why AI in Vocational Education and Training (VET)?
   - 1.2. Pedagogical Methodology: From Visual Intuition to Engineering Proficiency
   - 1.3. Architecture of the Moodle AI Suite (Filter, Quiz Question Type, TinyMCE Editor)
2. [Pedagogical Guide across the 5 Simulations](#2-pedagogical-guide-across-the-5-simulations)
   - 2.1. Module 1: Gradient Descent & Model Optimization (`gradient_descent`)
   - 2.2. Module 2: Confusion Matrix & Decision Threshold (`confusion_matrix`)
   - 2.3. Module 3: Polynomial Regression & Regularization (`regression`)
   - 2.4. Module 4: Non-linear Decision Boundaries – k-NN & Decision Tree (`knn_tree`)
   - 2.5. Module 5: Unsupervised Learning & K-Means Clustering (`kmeans`)
3. [Technical Instructions for Educators](#3-technical-instructions-for-educators)
   - 3.1. Inserting Simulations with the TinyMCE Visual Editor (`tiny_aisim`)
   - 3.2. Authoring Auto-Graded Interactive Assessment Tasks (`qtype_aisim`)
   - 3.3. Importing Ready-to-Use Question Banks (Moodle XML)
   - 3.4. Restoring Complete Pre-Configured Courses (`.mbz` Course Restore)
4. [Methodological Recommendations and Lesson Organization](#4-methodological-recommendations-and-lesson-organization)
   - 4.1. Structure of a 90-Minute Practical Workshop
   - 4.2. Formative Assessment and Error-Tolerant Learning
   - 4.3. Differentiated Instruction across Diverse Learner Profiles
5. [Appendix & Mathematical Reference Cheat Sheet](#5-appendix--mathematical-reference-cheat-sheet)

---

## 1. Introduction and Pedagogical Framework

### 1.1. The Challenge: Why AI in Vocational Education and Training (VET)?
In modern technical sectors—such as mechatronics, electrical engineering, industrial automation, healthcare technology, and IT—Artificial Intelligence (AI) and Machine Learning (ML) are no longer confined to graduate research. Today's vocational students frequently interface with:
- Automated computer-vision inspection systems on production assembly lines.
- Predictive condition monitoring of rotating machinery using vibration and temperature telemetry.
- Smart energy grid optimization and building climate management.
- Medical diagnostic classification and autonomous mobile robotic systems.

Yet traditional pedagogical approaches to machine learning often falter between two extremes:
1. **Mathematical Inaccessibility:** Excessive emphasis on multivariate calculus, partial differentials, and abstract matrix linear algebra that alienates applied learners.
2. **"Black-Box" Toolkits:** Superficial execution of Python scripts (`model.fit()`, `model.predict()`) where students observe predictions without grasping algorithmic dynamics.

### 1.2. Pedagogical Methodology: From Visual Intuition to Engineering Proficiency
The **AI for VET** plugin suite provides an **Interactive Inquiry-Based Learning (Active Learning)** framework:
- **Real-Time Visual Experimentation:** Learners manipulate hyperparameters using intuitive sliders and interactive buttons, immediately observing geometric and statistical responses on an HTML5 canvas at 60 FPS.
- **Zero-Install Client Execution:** Eliminates barriers associated with local Python installations, GPU drivers, or cloud servers. Simulations run entirely in the browser across desktops, tablets, and smartphones.
- **Dual Pedagogical Workflow:**
  1. *Exploration Phase (Lesson Pages / Books):* Free parameter exploration where learners develop intuition.
  2. *Application Phase (Quizzes):* Practical engineering challenges where students must guide the algorithm into predefined performance tolerances, with Moodle evaluating their work automatically.

### 1.3. Architecture of the Moodle AI Suite
The suite consists of three specialized plugins designed to operate seamlessly together:
- **`moodle-filter_aisim`** (Text Filter & Simulation Engines): Renders responsive HTML5 canvas widgets whenever `data-aisim` tags appear in content.
- **`moodle-qtype_aisim`** (Question Type & Auto-Grading): Embeds interactive tasks into Moodle Quizzes, evaluating student convergence criteria upon submission.
- **`moodle-tiny_aisim`** (TinyMCE Visual Editor Plugin): Provides a native modal dialog for teachers to insert, configure, and double-click edit simulation cards without writing HTML.

---

## 2. Pedagogical Guide across the 5 Simulations

### 2.1. Module 1: Gradient Descent & Model Optimization (`gradient_descent`)
- **Key Concepts:** Cost function $J(w)$, learning rate $\alpha$, convex vs. non-convex landscapes, global vs. local minima.
- **Pedagogical Metaphor:** A hiker navigating down a dense foggy mountain into a valley. The hiker cannot see the map, only the slope beneath their feet. The step size taken in the steepest downward direction represents the **learning rate ($\alpha$)**.
- **Practical Application:** Tuning feedback controllers in mechatronics; backpropagation in neural networks.
- **Key Learning Observations:**
  1. *Small Learning Rate ($\alpha = 0.01$):* Convergence is exceedingly sluggish.
  2. *Optimal Learning Rate ($\alpha = 0.10 - 0.20$):* Fast, steady descent directly to the minimum.
  3. *Excessive Learning Rate ($\alpha \ge 1.05$):* Severe oscillations and mathematical divergence.
  4. *Non-Convex Surfaces:* Demonstrates how initial starting weights ($w_0$) determine which local minimum the model lands in.

---

### 2.2. Module 2: Confusion Matrix & Decision Threshold (`confusion_matrix`)
- **Key Concepts:** Binary classification evaluation, imbalanced datasets, decision threshold $\theta$, TP, FP, FN, TN, Precision, Recall, F1-score.
- **Why Accuracy Can Be Misleading:** In quality inspection where only 2% of manufactured units are defective, a naive classifier that flags all units as flawless achieves a misleading **98% Accuracy** while failing to catch any defects.
- **Metric Definitions:**
  - *Threshold ($\theta$):* Probability cutoff above which an item is classified as defective.
  - *Precision:* Percentage of flagged items that are genuinely defective (avoids false alarms).
  - *Recall:* Percentage of all true defects caught by the model (avoids missed defects).
  - *F1-score:* The harmonic mean balancing Precision and Recall.
- **Interactive Dynamics:** Moving threshold $\theta$ to the left increases Recall at the cost of Precision; moving it right increases Precision at the cost of missed defects.

---

### 2.3. Module 3: Polynomial Regression & Regularization (`regression`)
- **Key Concepts:** Non-linear modeling, polynomial degree $d$, underfitting, overfitting, generalization gap, $L_2$ Ridge regularization ($\lambda$).
- **Pedagogical Metaphor:** Underfitting is skimming only chapter headings; overfitting is memorizing every typographical quirk in a textbook. Regularization ($\lambda$) enforces a simplicity constraint that prevents wild oscillations.
- **Key Learning Observations:**
  - $d=1$: Line cannot capture sinusoidal curves (high training and test error).
  - $d=3$: Smooth curve fitting the underlying physical signal.
  - $d=5$ with $\lambda=0$: Curve passes through every noise point, resulting in severe boundary spikes.
  - Adding $\lambda=0.01-0.10$: Penalizes extreme weights and restores smooth generalization.

---

### 2.4. Module 4: Non-linear Decision Boundaries – k-NN & Decision Tree (`knn_tree`)
- **Key Concepts:** Instance-based learning (k-NN) versus hierarchical partitioning (Decision Tree), boundary morphology, hyperparameter $k$, tree max depth.
- **Comparison:**
  - *k-NN:* Assigns classes by majority vote among nearest neighbors, producing smooth organic contours.
  - *Decision Tree:* Partitions space using axis-parallel rectangular boundaries.
- **Hyperparameter Effects:** Low $k=1$ is fragile to outliers; large $k=15$ oversmooths. Shallow trees underfit; deep trees fragment the space into tiny overfitting partitions.

---

### 2.5. Module 5: Unsupervised Learning & K-Means Clustering (`kmeans`)
- **Key Concepts:** Unlabeled grouping, centroids, Euclidean distance, inertia (within-cluster sum of squares), iterative convergence.
- **Interactive Step Dynamics:** Students click "Step" to observe the alternating loop:
  1. *Assignment step:* Points take the color of their nearest centroid.
  2. *Update step:* Centroids shift to the geometric center of their assigned cluster.
- **Hyperparameter Effects:** Setting $K=2$ on a 3-cluster dataset demonstrates underclustering, showing how inertia levels off at the true cluster count.

---

## 3. Technical Instructions for Educators

### 3.1. Inserting Simulations with the TinyMCE Visual Editor (`tiny_aisim`)
1. In the TinyMCE editor toolbar, click the **AI Simulation icon** (or choose *Insert > AI Simulation...*).
2. Configure parameters:
   - Select the desired simulation engine.
   - Adjust domain scenario, threshold, learning rate, degree, or cluster count.
   - Check **Read-only mode** if you wish to lock controls for guided observation.
   - Inspect the live simulation card preview at the bottom of the modal.
3. Click **Insert Simulation**.
4. **Editing existing simulations:** Simply **double-click** any simulation card directly in the editor to reopen the dialog pre-filled with its configuration.

---

### 3.2. Authoring Auto-Graded Interactive Assessment Tasks (`qtype_aisim`)
1. Inside your Moodle Quiz, select **Questions > Add a new question**.
2. Select **AI Simulation (`aisim`)**.
3. Specify question name, problem statement, default mark, and penalty.
4. Under **AI Simulation Settings**:
   - Choose the simulation type and scenario.
   - Set target metrics (e.g. Min = `0.85`, Max = `1.00`, Tolerance = `0.05`).
   - Define custom automated feedback strings for full, partial, and incorrect answers.
5. Save the question. During quiz attempts, Moodle tracks student parameter interactions and automatically calculates awarded points upon submission.

---

### 3.3. Importing Ready-to-Use Question Banks (Moodle XML)
1. Download [`questions_aiforvet.xml`](file:///C:/tsp/moodle-qtype_aisim/demo/questions_aiforvet.xml).
2. In your course, navigate to **Question bank > Import**.
3. Select **Moodle XML format**.
4. Upload `questions_aiforvet.xml` and click **Import**.
5. The full question set is immediately available in your course question bank.

---

### 3.4. Restoring Complete Pre-Configured Courses (`.mbz` Course Restore)
1. Download the official course archive from the AI for VET repository.
2. In Moodle, navigate to **Site administration > Courses > Restore course**.
3. Upload the `.mbz` file and complete the restore wizard to generate an instant turn-key course.

---

## 4. Methodological Recommendations and Lesson Organization

### 4.1. Structure of a 90-Minute Practical Workshop
```
[00-15 min] Real-World Problem Presentation & Industrial Context
     │
     ▼
[15-35 min] Guided Visual Exploration via Simulation Lesson Pages
     │
     ▼
[35-50 min] Pair Discussion, Hypothesis Testing & Metric Formulation
     │
     ▼
[50-75 min] Individual Problem Solving in Auto-Graded Moodle Quizzes
     │
     ▼
[75-90 min] Reflection, Synthesis & Vocational Domain Mapping
```

### 4.2. Formative Assessment and Error-Tolerant Learning
- Configure quiz attempts to **Immediate Feedback** or **Interactive with Multiple Tries**.
- Encourage iterative experimentation: students should not be penalized for testing boundary conditions, but rewarded for identifying why divergence or misclassification occurred.

### 4.3. Differentiated Instruction across Diverse Learner Profiles
- **Foundation Track:** Focus on graphical sliders, verifying whether target badges turn green.
- **Advanced Engineering Track:** Analyze mathematical trade-offs between precision and recall, discuss matrix condition numbers, and compare regularized loss surfaces.

---

## 5. Appendix & Mathematical Reference Cheat Sheet

| Simulation Name | Key Parameters | Model / Governing Equation | Target Success Metric |
| :--- | :--- | :--- | :--- |
| **Gradient Descent** | $\alpha$ (step), $w_0$ (start) | $w \leftarrow w - \alpha \nabla J(w)$ | $J(w) \le 0.05$ in $< 30$ iterations |
| **Confusion Matrix** | $\theta$ (decision threshold) | $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | $F_1 \ge 0.85$ |
| **Polynomial Regression** | $d$ (degree), $\lambda$ (regularization) | $J(w) = \frac{1}{2m}\sum (h_w(x) - y)^2 + \frac{\lambda}{2}\sum w_j^2$ | Test $R^2 \ge 0.85$ |
| **k-NN Classification** | $k$ (neighbors) | $\hat{y} = \arg\max_c \sum_{i \in N_k(x)} I(y_i = c)$ | Accuracy $\ge 88\%$ |
| **K-Means Clustering** | $K$ (clusters) | $\min_{\mu} \sum_{i=1}^m \sum_{k=1}^K r_{ik} \|x^{(i)} - \mu_k\|^2$ | Stable Centroid Convergence |

---

*This guide was developed under the Erasmus+ AI for VET project and is released as an Open Educational Resource (OER) for all educators and training institutions.*
