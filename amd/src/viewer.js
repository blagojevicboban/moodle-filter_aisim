// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * AI Simulation viewer AMD module.
 *
 * Scans the page for [data-aisim] elements and mounts the corresponding simulation engine.
 *
 * @module filter_aisim/viewer
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {GradientDescentSim} from 'filter_aisim/sims/gradient_descent';
import {ConfusionMatrixSim} from 'filter_aisim/sims/confusion_matrix';
import {RegressionSim} from 'filter_aisim/sims/regression';
import {KnnTreeSim} from 'filter_aisim/sims/knn_tree';
import {KMeansSim} from 'filter_aisim/sims/kmeans';
import {NeuralNetworkSim} from 'filter_aisim/sims/neural_network';
import {ConvolutionSim} from 'filter_aisim/sims/convolution';
import {QLearningSim} from 'filter_aisim/sims/q_learning';
import {DataBiasSim} from 'filter_aisim/sims/data_bias';
import {LogisticSim} from 'filter_aisim/sims/logistic';
import {eventTypes} from 'core_filters/events';
import Notification from 'core/notification';
import {load as loadStrings} from 'filter_aisim/strings';

const SELECTOR = '[data-aisim]';
let initialised = false;

/**
 * Read a data-* attribute, falling back to a default when it is missing or empty.
 *
 * @param {HTMLElement} el
 * @param {String} name Attribute name without the data- prefix
 * @param {*} fallback
 * @returns {*}
 */
const attr = (el, name, fallback) => el.getAttribute('data-' + name) || fallback;

/**
 * A simulation engine with the function that reads its options from the element.
 *
 * @param {Function} Engine Simulation class
 * @param {Function} options (el) => options object
 * @returns {Object}
 */
const sim = (Engine, options) => ({Engine, options});

/**
 * Create a k-NN / decision tree simulation; data-model ('knn' or 'tree') overrides the initial model.
 *
 * @param {String} model 'knn' or 'tree'
 * @returns {Object}
 */
const knnTree = (model) => sim(KnnTreeSim, (el) => ({
    model: attr(el, 'model', model),
    dataset: attr(el, 'dataset', 'moons'),
    k: attr(el, 'k', 3),
    depth: attr(el, 'depth', 3)
}));

const gradientDescent = sim(GradientDescentSim, (el) => ({
    profile: attr(el, 'profile', 'convex'),
    lr: attr(el, 'lr', ''),
    w0: attr(el, 'w0', '')
}));

const confusionMatrix = sim(ConfusionMatrixSim, (el) => ({
    scenario: attr(el, 'scenario', 'quality'),
    threshold: attr(el, 'threshold', 0.50)
}));

const regression = sim(RegressionSim, (el) => ({
    dataset: attr(el, 'dataset', 'sine'),
    degree: attr(el, 'degree', 3),
    lambda: attr(el, 'lambda', 0.0)
}));

const kmeans = sim(KMeansSim, (el) => ({
    dataset: attr(el, 'dataset', 'blobs3'),
    k: attr(el, 'k', 3)
}));

const neuralNetwork = sim(NeuralNetworkSim, (el) => ({
    dataset: attr(el, 'dataset', 'xor'),
    layers: attr(el, 'layers', 0),
    neurons: attr(el, 'neurons', 4),
    activation: attr(el, 'activation', 'tanh'),
    lr: attr(el, 'lr', 0.3),
    // Same seed and epochs give the same trained network (an embedded, already trained example).
    seed: attr(el, 'seed', 1),
    epochs: attr(el, 'epochs', 0)
}));

const convolution = sim(ConvolutionSim, (el) => ({
    image: attr(el, 'image', 'shapes'),
    // A preset name, or the 9 weights as "a,b,c,...", with an optional divisor and absolute value.
    preset: attr(el, 'preset', 'identity'),
    kernel: attr(el, 'kernel', ''),
    divisor: attr(el, 'divisor', ''),
    abs: attr(el, 'abs', undefined),
    target: attr(el, 'target', ''),
    presets: attr(el, 'presets', true)
}));

const qLearning = sim(QLearningSim, (el) => ({
    map: attr(el, 'map', 'maze'),
    alpha: attr(el, 'alpha', 0.5),
    gamma: attr(el, 'gamma', 0.9),
    epsilon: attr(el, 'epsilon', 0.2),
    // Same seed and episodes give the same Q-table (an embedded, already trained agent).
    seed: attr(el, 'seed', 1),
    episodes: attr(el, 'episodes', 0)
}));

const dataBias = sim(DataBiasSim, (el) => ({
    population: attr(el, 'population', 'shifted'),
    n: attr(el, 'n', 200),
    share: attr(el, 'share', 5),
    test: attr(el, 'test', 25),
    k: attr(el, 'k', 5),
    seed: attr(el, 'seed', 1)
}));

const logistic = sim(LogisticSim, (el) => ({
    dataset: attr(el, 'dataset', 'machine'),
    w1: attr(el, 'w1', 0),
    w2: attr(el, 'w2', 0),
    b: attr(el, 'b', 0),
    threshold: attr(el, 'threshold', 0.5),
    lr: attr(el, 'lr', 0.5)
}));

/** Simulation factories by data-aisim value (several aliases per simulation). */
const SIMS = {
    'gradient-descent': gradientDescent,
    'gradient_descent': gradientDescent,
    'confusion-matrix': confusionMatrix,
    'confusion_matrix': confusionMatrix,
    'regression': regression,
    'polynomial-regression': regression,
    'knn-tree': knnTree('knn'),
    // The editor button writes the underscore form.
    'knn_tree': knnTree('knn'),
    'knn': knnTree('knn'),
    'decision-tree': knnTree('tree'),
    'kmeans': kmeans,
    'k-means': kmeans,
    'neural-network': neuralNetwork,
    'neural_network': neuralNetwork,
    'perceptron': neuralNetwork,
    'convolution': convolution,
    'image-filters': convolution,
    'q-learning': qLearning,
    'q_learning': qLearning,
    'reinforcement-learning': qLearning,
    'data-bias': dataBias,
    'data_bias': dataBias,
    'data-split': dataBias,
    'logistic': logistic,
    'logistic-regression': logistic,
    'logistic_regression': logistic
};

/**
 * Mount simulations on elements matching SELECTOR.
 *
 * @param {HTMLElement|Document} root
 */
const mount = (root) => {
    if (!root || !root.querySelectorAll) {
        return;
    }
    root.querySelectorAll(SELECTOR).forEach((el) => {
        // Prevent double mounting.
        if (el.dataset.aisimMounted) {
            return;
        }
        el.dataset.aisimMounted = 'true';

        const simType = el.getAttribute('data-aisim');
        if (Object.prototype.hasOwnProperty.call(SIMS, simType)) {
            const {Engine, options} = SIMS[simType];
            loadStrings(Engine.STRINGS)
                // The editor button marks a fixed example with data-readonly="1": the controls are locked.
                .then((strings) => new Engine(el, {...options(el), readonly: attr(el, 'readonly', '0') === '1', strings}))
                .catch(Notification.exception);
        }
    });
};

/**
 * Initialize the viewer.
 */
export const init = () => {
    mount(document);
    if (!initialised) {
        initialised = true;
        // Listen for filter content updated event (e.g., in quiz page transitions or AJAX).
        document.addEventListener(eventTypes.filterContentUpdated, (e) => {
            mount(e.detail && e.detail.nodes ? e.detail.nodes[0] : document);
        });
    }
};
