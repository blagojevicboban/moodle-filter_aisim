# AI for VET – Pedagogical and Technical Guide for Teachers
## Integrating Interactive AI and Machine Learning Simulations into Moodle

*Project: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Programme: **Erasmus+***  
*Official Portal:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Simulations without Moodle:* [https://sim.aiforvet.eu/](https://sim.aiforvet.eu/)  
*Authors and Technical Development Team of AI for VET*  
*Guide Version: 1.1 (October 2026)*  
*Compatibility: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Table of Contents
1. [Introduction and Pedagogical Framework](#1-introduction-and-pedagogical-framework)
   - 1.1. The Challenge: Why AI in Vocational Education and Training (VET)?
   - 1.2. Pedagogical Methodology: From Visual Intuition to Engineering Proficiency
   - 1.3. Architecture of the Moodle AI Suite (Filter, Quiz Question Type, TinyMCE Editor)
2. [Pedagogical Guide across the 9 Simulations](#2-pedagogical-guide-across-the-9-simulations)
   - 2.1. Module 1: Gradient Descent & Model Optimization (`gradient_descent`)
   - 2.2. Module 2: Confusion Matrix & Decision Threshold (`confusion_matrix`)
   - 2.3. Module 3: Polynomial Regression & Regularization (`regression`)
   - 2.4. Module 4: Non-linear Decision Boundaries – k-NN & Decision Tree (`knn_tree`)
   - 2.5. Module 5: Unsupervised Learning & K-Means Clustering (`kmeans`)
   - 2.6. Module 6: Perceptron & Neural Network (`neural_network`)
   - 2.7. Module 7: Convolution & Image Filters (`convolution`)
   - 2.8. Module 8: Reinforcement Learning – Q-learning (`q_learning`)
   - 2.9. Module 9: Data Split & Bias (`data_bias`)
3. [Technical Instructions for Educators](#3-technical-instructions-for-educators)
   - 3.1. Inserting Simulations with the TinyMCE Visual Editor (`tiny_aisim`)
   - 3.2. Authoring Auto-Graded Interactive Assessment Tasks (`qtype_aisim`)
   - 3.3. Importing the Ready-to-Use Question Bank (Moodle XML)
   - 3.4. AISim Lab: the Simulations without Moodle
4. [Methodological Recommendations and Lesson Organization](#4-methodological-recommendations-and-lesson-organization)
   - 4.1. Structure of a 90-Minute Practical Workshop
   - 4.2. Formative Assessment and Error-Tolerant Learning
   - 4.3. Differentiated Instruction across Diverse Learner Profiles
   - 4.4. Suggested Order of the Simulations
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
- **Real-Time Visual Experimentation:** Learners manipulate hyperparameters using intuitive sliders and buttons and immediately see how the algorithm responds.
- **Zero-Install Client Execution:** Eliminates barriers associated with local Python installations, GPU drivers, or cloud servers. Simulations run entirely in the browser across desktops, tablets, and smartphones.
- **Dual Pedagogical Workflow:**
  1. *Exploration Phase (Lesson Pages / Books):* Free parameter exploration where learners develop intuition.
  2. *Application Phase (Quizzes):* Practical engineering challenges where students must guide the algorithm into predefined performance tolerances, with Moodle evaluating their work automatically.

### 1.3. Architecture of the Moodle AI Suite
The suite consists of three specialized plugins designed to operate seamlessly together:
- **`moodle-filter_aisim`** (Text Filter & Simulation Engines): Renders the 9 interactive simulations whenever `data-aisim` tags appear in content.
- **`moodle-qtype_aisim`** (Question Type & Auto-Grading): Embeds interactive tasks into Moodle Quizzes, evaluating student convergence criteria upon submission.
- **`moodle-tiny_aisim`** (TinyMCE Visual Editor Plugin): Provides a native modal dialog for teachers to insert, configure, and double-click edit simulation cards without writing HTML.

The simulations follow the user's Moodle language: English, Serbian (Latin and Cyrillic), Slovak and Portuguese are included.

---

## 2. Pedagogical Guide across the 9 Simulations

The simulations go from simpler to more demanding. The first five cover the foundations (optimization, model evaluation, overfitting, classification, clustering); modules 6–9 build on them with neural networks, image processing, learning from rewards and the responsible use of data.

### 2.1. Module 1: Gradient Descent & Model Optimization (`gradient_descent`)
- **Key Concepts:** Cost function J(w), learning rate α, convex vs. non-convex landscapes, global vs. local minima.
- **Pedagogical Metaphor:** A hiker navigating down a dense foggy mountain into a valley. The hiker cannot see the map, only the slope beneath their feet. The step size taken in the steepest downward direction represents the **learning rate (α)**.
- **Practical Application:** Tuning feedback controllers in mechatronics; backpropagation in neural networks.
- **Key Learning Observations:**
  1. *Small Learning Rate (α = 0.01):* Convergence is exceedingly sluggish.
  2. *Optimal Learning Rate (α = 0.10 – 0.20):* Fast, steady descent directly to the minimum.
  3. *Excessive Learning Rate (α ≥ 1.05):* Severe oscillations and mathematical divergence.
  4. *Non-Convex Surfaces:* Demonstrates how the initial starting weight (w₀) determines which local minimum the model lands in.
- **Typical Quiz Task:**
  > *"The convex cost function has its minimum at J(w) = 1.00. Set the learning rate (α) and run the algorithm so that the cost drops to J(w) ≤ 1.01."*

---

### 2.2. Module 2: Confusion Matrix & Decision Threshold (`confusion_matrix`)
- **Key Concepts:** Binary classification evaluation, imbalanced datasets, decision threshold θ, TP, FP, FN, TN, Precision, Recall, F1-score.
- **Why Accuracy Can Be Misleading:** In quality inspection where only 2% of manufactured units are defective, a naive classifier that flags all units as flawless achieves a misleading **98% Accuracy** while failing to catch any defects.
- **Metric Definitions:**
  - *Threshold (θ):* Probability cutoff above which an item is classified as defective.
  - *Precision:* Percentage of flagged items that are genuinely defective (avoids false alarms).
  - *Recall:* Percentage of all true defects caught by the model (avoids missed defects).
  - *F1-score:* The harmonic mean balancing Precision and Recall.
- **Interactive Dynamics:** Moving threshold θ to the left increases Recall at the cost of Precision; moving it right increases Precision at the cost of missed defects.
- **Typical Quiz Task:**
  > *"Set the decision threshold (θ) so that the classifier catches most defects without too many false alarms: Recall ≥ 0.80 and Precision ≥ 0.70."*

---

### 2.3. Module 3: Polynomial Regression & Regularization (`regression`)
- **Key Concepts:** Non-linear modeling, polynomial degree d, underfitting, overfitting, generalization gap, L₂ Ridge regularization (λ).
- **Pedagogical Metaphor:** Underfitting is skimming only chapter headings; overfitting is memorizing every typographical quirk in a textbook. Regularization (λ) enforces a simplicity constraint that prevents wild oscillations.
- **Key Learning Observations:**
  - d = 1: A line cannot capture sinusoidal curves (high training and test error).
  - d = 3: Smooth curve fitting the underlying physical signal.
  - d = 5 with λ = 0: The curve passes through every noise point, resulting in severe boundary spikes.
  - Adding λ = 0.01 – 0.10: Penalizes extreme weights and restores smooth generalization.
- **Typical Quiz Task:**
  > *"For a noisy periodic signal, choose a polynomial degree of at most 7 so that the model generalizes the trend without overfitting, with a validation error MSE ≤ 0.19."*

---

### 2.4. Module 4: Non-linear Decision Boundaries – k-NN & Decision Tree (`knn_tree`)
- **Key Concepts:** Instance-based learning (k-NN) versus hierarchical partitioning (Decision Tree), boundary morphology, hyperparameter k, tree max depth.
- **Comparison:**
  - *k-NN:* Assigns classes by majority vote among nearest neighbors, producing smooth organic contours.
  - *Decision Tree:* Partitions space using axis-parallel rectangular boundaries.
- **Hyperparameter Effects:** Low k = 1 is fragile to outliers; large k = 15 oversmooths. Shallow trees underfit; deep trees fragment the space into tiny overfitting partitions.

---

### 2.5. Module 5: Unsupervised Learning & K-Means Clustering (`kmeans`)
- **Key Concepts:** Unlabeled grouping, centroids, Euclidean distance, inertia (within-cluster sum of squares), iterative convergence.
- **Interactive Step Dynamics:** Students click "Next Step" to observe the alternating loop:
  1. *Assignment step:* Points take the color of their nearest centroid.
  2. *Update step:* Centroids shift to the geometric center of their assigned cluster.
- **Hyperparameter Effects:** Setting K = 2 on a 3-cluster dataset demonstrates underclustering, showing how inertia levels off at the true cluster count.

---

### 2.6. Module 6: Perceptron & Neural Network (`neural_network`)
- **Key Concepts:** Perceptron, linear separability, hidden layer, activation functions (sigmoid, tanh, ReLU), epochs, loss (log-loss), learning rate, initial weights.
- **Pedagogical Metaphor:** A single neuron is a ruler: it can draw only one straight line and say "red on this side, blue on that side". A hidden layer is a team of rulers whose lines the output neuron combines into a bent boundary – like cutting a circle out of paper with several straight cuts. More neurons, finer cuts.
- **Practical Application:** Telling good parts from defective ones using several sensor readings. Neural networks underpin speech and image recognition and large language models – here the same idea appears in its smallest form.
- **Key Learning Observations:**
  1. *Linearly separable data, 0 hidden layers:* The perceptron quickly learns a straight line and reaches almost 100%.
  2. *XOR (crossed quadrants), 0 hidden layers:* Accuracy stays around 50% however long it trains – no straight line separates the classes. The simulation's advice says so.
  3. *XOR with one hidden layer of 2–4 neurons:* The boundary bends and accuracy passes 98%. Line widths in the network drawing show which connections have become strong.
  4. *Circle in a circle:* With 2 neurons the network often stalls; with 3–4 neurons and "New weights" it finds a closed boundary around the inner class.
  5. *Too high a learning rate (e.g. 3):* The loss explodes – the same lesson as in gradient descent, because training a network *is* gradient descent.
- **Typical Quiz Task:**
  > *"On the XOR data a perceptron stays at 50% accuracy. Add a hidden layer and reach accuracy ≥ 98% with at most 4 hidden neurons."*

---

### 2.7. Module 7: Convolution & Image Filters (`convolution`)
- **Key Concepts:** An image as a table of numbers, 3×3 kernel, divisor, blur, sharpen, edge detection (Sobel, Laplacian), absolute value, convolutional neural networks (CNN).
- **Pedagogical Metaphor:** The kernel is a small "magnifier with a rule" sliding over the image. At every position it looks at 9 pixels, multiplies them by its weights and writes down one number. If all weights are equal, the magnifier averages (blurs). If it subtracts one side from the other, it only "sees" where the brightness changes sharply – the edges.
- **Practical Application:** Optical quality inspection (scratches, cracks and part edges on the production line); removing noise from camera and medical images. The first layers of every image-recognition network (CNN) learn exactly such kernels.
- **Key Learning Observations:**
  1. *Tapping a pixel* shows the whole calculation: 9 values, 9 weights, the sum and the division. This is the key moment of the lesson – convolution stops being "magic".
  2. *Blur (average) and Gaussian blur:* The noise disappears, but the edges soften too.
  3. *Sum of weights / divisor ≠ 1:* The whole image gets brighter or darker – the simulation warns about it.
  4. *Sobel without the absolute value:* Edges from light to dark give negative values and show as black. With |x| both kinds of edge appear.
  5. *Target image and match (R²):* Students guess or build the filter that produces a given result.
- **Typical Quiz Task:**
  > *"The image is full of noise and the ready-made filters are hidden. Type kernel weights and a divisor so that the output looks like the target, with a match ≥ 95%."* (One solution: weights 1 2 1 / 2 4 2 / 1 2 1 and divisor 16.)

---

### 2.8. Module 8: Reinforcement Learning – Q-learning (`q_learning`)
- **Key Concepts:** Agent, state, action, reward, Q-table, episode, learning rate α, discount factor γ, exploration ε (exploration vs. exploitation), policy.
- **Pedagogical Metaphor:** Training a dog: nobody explains the rules, the dog just gets a treat when it does the right thing. After many attempts it remembers which move in which situation pays off. The agent in the simulation does the same: −0.1 for every step, +10 at the goal, −10 for falling off the cliff.
- **Practical Application:** Robot and autonomous-vehicle control (moving, grasping); energy management (battery charging, heating) and traffic lights; game AI from chess and Go to video games.
- **Key Learning Observations:**
  1. *Maze:* At first the agent wanders at random. After a few dozen episodes the cell colours and arrows form a "signpost" to the goal and the learned path shrinks to 10 steps.
  2. *Cliff with a large ε (e.g. 0.4):* The agent learns the shortest path right along the edge but keeps falling off at random during training – the average reward is low. With ε = 0.05 it falls far less often.
  3. *Near or far with γ = 0.5:* The agent takes the small treasure T (+3) because the future is worth little to it. Raising γ (e.g. 0.9) makes the big goal G (+10) more valuable – but only if the agent has discovered it at all, which needs enough ε.
  4. *Tapping a cell* shows the Q-values of all four moves – students see the numbers behind the arrow.
- **Typical Quiz Task:**
  > *"The agent picks the small treasure T (+3) nearby. Change the discount factor γ and the exploration ε so that it learns to go to the big goal G (+10) in 4 steps, within 1000 episodes. Explain the role of each parameter."*

---

### 2.9. Module 9: Data Split & Bias (`data_bias`)
- **Key Concepts:** Training and test sets, generalization, memorisation (k = 1), representative samples, data bias, per-group accuracy, model fairness.
- **Pedagogical Metaphor:** An exam made only of exercises the student has already seen says nothing. And if both the exercises and the exam come from a textbook without a single geometry problem, the student still gets top marks – and fails the first time they see a triangle. A test set can only reveal the errors its sample contains.
- **Practical Application:** A quality-control model trained on parts from one factory and used in another (a different machine, a different material). Credit, hiring or medical-diagnosis systems that work worse for groups that were rare in the data. The EU AI Act asks for exactly this: high-quality, representative data and checks across groups.
- **Key Learning Observations:**
  1. *Factory B share 5%:* The test set reports about 95%, while the real-world accuracy for factory B is far lower. The average hides the rare group.
  2. *Raising the share of B (e.g. to 30–50%):* Accuracy for B jumps although the total sample size stays the same – *which* data matters more than *how much*.
  3. *"B behaves similarly":* Under-representation does no harm when the groups do not differ – the problem appears when they differ and are not represented.
  4. *k = 1:* Training accuracy is 100% (the model has memorised every point), which says nothing about its quality.
  5. *A small test set:* The test score jumps with every new sample (🎲) – a small test is an unreliable measure.
- **Typical Quiz Task:**
  > *"Collecting data is expensive. With at most 120 samples, get the real-world accuracy of both factories to ≥ 88%. What matters more: how much data or which data?"*

---

## 3. Technical Instructions for Educators

### 3.1. Inserting Simulations with the TinyMCE Visual Editor (`tiny_aisim`)
1. In the TinyMCE editor toolbar, click the **AI Simulation icon** (or choose *Insert > AI Simulation...*).
2. Configure parameters:
   - Select one of the 9 simulations.
   - Adjust the scenario, threshold, learning rate, degree, cluster count, network architecture, image and filter, agent map or the group share in the data.
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
   - Set Target Metric 1 and 2 (e.g. recall `0.80` and precision `0.70` for the confusion matrix). Their meaning depends on the simulation – see the help (?) next to the fields and the table in the Appendix.
   - The tolerance is used for gradient descent (half marks up to target + tolerance) and convolution (half marks down to target − tolerance).
   - Define custom automated feedback strings for full, partial, and incorrect answers.
5. Save the question. During quiz attempts, Moodle tracks student parameter interactions and automatically calculates awarded points upon submission.

---

### 3.3. Importing the Ready-to-Use Question Bank (Moodle XML)
The `moodle-qtype_aisim` repository contains 27 ready-made tasks – three for each of the 9 simulations (basic, intermediate and advanced level):
1. Download [`questions_aiforvet.xml`](https://github.com/blagojevicboban/moodle-qtype_aisim/raw/master/demo/questions_aiforvet.xml).
2. In your course, navigate to **Question bank > Import**.
3. Select **Moodle XML format**.
4. Upload `questions_aiforvet.xml` and click **Import**.
5. The category *"AI za VET - Baza simulacionih zadataka (27 pitanja)"* is created with all questions ready to add to any quiz. (The question texts are in Serbian; edit them as needed.)

---

### 3.4. AISim Lab: the Simulations without Moodle
At [sim.aiforvet.eu](https://sim.aiforvet.eu/) all 9 simulations run on their own – no account, no data collected – in English, Serbian, Slovak and Portuguese. Each simulation has a short explanation and three challenges.
- **Homework and phones:** Students can practise outside Moodle; challenge progress is kept only in their own browser.
- **"Copy link to this setup":** Creates a link to exactly this state of the simulation – handy for the projector or for a message to students.
- **"Embed in Moodle":** Gives ready-made HTML for that setup to paste into any Moodle text (needs `filter_aisim`).

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
- The advanced bank tasks (e.g. Q-learning and data bias) end with an "Explain…" question – use it for a short oral or written answer after the quiz.

### 4.3. Differentiated Instruction across Diverse Learner Profiles
- **Foundation Track:** Focus on graphical sliders, verifying whether target badges turn green. In convolution students pick ready-made filters; in Q-learning they watch the agent learn the maze.
- **Advanced Engineering Track:** Analyze the trade-off between precision and recall, compare regularized fits, build kernels by hand, look for the smallest neural network that solves a task, and explain how γ and ε change the agent's behaviour.

### 4.4. Suggested Order of the Simulations
| Lesson / topic | Simulations | Key idea |
| :--- | :--- | :--- |
| 1. How a model learns | Gradient Descent | Learning = reducing the error in small steps |
| 2. How we measure a model | Confusion Matrix, Data Split & Bias | Accuracy is not enough; the test must be fair and representative |
| 3. Model complexity | Polynomial Regression, k-NN & Decision Tree | Too simple vs. overfitted |
| 4. Learning without answers | K-Means | Grouping without labels |
| 5. Neural networks and images | Perceptron & Neural Network, Convolution | From one neuron to image recognition |
| 6. Learning from rewards | Q-learning | Exploration vs. exploitation, the value of the future |

---

## 5. Appendix & Mathematical Reference Cheat Sheet

| Simulation Name | Key Parameters | Model / Governing Equation | Target Success Metric |
| :--- | :--- | :--- | :--- |
| **Gradient Descent** | α (step), w₀ (start) | w ← w − α · J′(w) | J(w) ≤ 1.01 (the minimum is 1.00) |
| **Confusion Matrix** | θ (decision threshold) | F1 = 2 · P · R / (P + R) | Recall ≥ 0.80 and Precision ≥ 0.70 |
| **Polynomial Regression** | d (degree), λ (regularization) | J(w) = (1/2m) · Σ(h(x) − y)² + (λ/2) · Σw² | Validation MSE ≤ 0.19 with d ≤ 7 |
| **k-NN Classification** | k (neighbors) | ŷ = most common class among the k nearest neighbours | Accuracy ≥ 98% |
| **K-Means Clustering** | K (clusters) | min Σ ‖x − μ‖² (sum of squared distances to the centroids) | Convergence with K = 3 and WCSS ≤ 10 |
| **Neural Network** | layers, neurons, activation, learning rate | a = f(w · x + b); w ← w − α · ∂L/∂w | Accuracy ≥ 98% with at most 4 hidden neurons |
| **Convolution** | 3×3 kernel, divisor, \|x\| | y = Σ (weight · pixel) / divisor | Match with the target (R²) ≥ 95% |
| **Q-learning** | α, γ, ε, episodes | Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,·) − Q(s,a)) | Shortest path (e.g. 10 steps) within ≤ 200 episodes |
| **Data Split & Bias** | sample size, share of group B, test %, k | worst-group accuracy = min(accuracy A, accuracy B) | Both factories ≥ 88% with ≤ 120 samples |

#### Meaning of the "Target Metric" fields in questions (`qtype_aisim`)
| Simulation | Target Metric 1 | Target Metric 2 | Half marks |
| :--- | :--- | :--- | :--- |
| Gradient Descent | highest allowed cost J(w) | – | cost up to target + tolerance |
| Confusion Matrix | minimum Recall | minimum Precision | one condition met |
| Polynomial Regression | highest validation MSE | highest degree d | one condition met |
| k-NN & Decision Tree | minimum test accuracy | – | accuracy up to 5% below target |
| K-Means | highest inertia (WCSS) | exact number of clusters K (0 = any) | one condition met |
| Neural Network | minimum accuracy | maximum hidden neurons (0 = no limit) | one condition met |
| Convolution | minimum match (R²) | – | match down to target − tolerance |
| Q-learning | maximum steps to the goal | maximum training episodes (0 = no limit) | the agent reaches the goal, but by a longer path or after more episodes |
| Data Split & Bias | minimum accuracy of the weakest factory | maximum sample size (0 = no limit) | accuracy met, but with too many samples |

---

*This guide was developed under the Erasmus+ AI for VET project and is released as an Open Educational Resource (OER) for all educators and training institutions.*
