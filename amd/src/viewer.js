/**
 * AI Simulation viewer AMD module.
 *
 * Scans the page for [data-aisim] elements and mounts the corresponding simulation engine.
 *
 * @module filter_aisim/viewer
 * @copyright 2026 Boban Blagojevic
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
import {eventTypes} from 'core_filters/events';

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
 * Create a k-NN / decision tree simulation with the given initial model.
 *
 * @param {String} model 'knn' or 'tree'
 * @returns {Function}
 */
const knnTree = (model) => (el, lang) => new KnnTreeSim(el, {
    lang: lang,
    model: model,
    dataset: attr(el, 'dataset', 'moons'),
    k: attr(el, 'k', 3),
    depth: attr(el, 'depth', 3)
});

const gradientDescent = (el, lang) => new GradientDescentSim(el, {
    lang: lang,
    profile: attr(el, 'profile', 'convex'),
    lr: attr(el, 'lr', ''),
    w0: attr(el, 'w0', '')
});

const confusionMatrix = (el, lang) => new ConfusionMatrixSim(el, {
    lang: lang,
    scenario: attr(el, 'scenario', 'quality'),
    threshold: attr(el, 'threshold', 0.50)
});

const regression = (el, lang) => new RegressionSim(el, {
    lang: lang,
    dataset: attr(el, 'dataset', 'sine'),
    degree: attr(el, 'degree', 3),
    lambda: attr(el, 'lambda', 0.0)
});

const kmeans = (el, lang) => new KMeansSim(el, {
    lang: lang,
    dataset: attr(el, 'dataset', 'blobs3'),
    k: attr(el, 'k', 3)
});

const neuralNetwork = (el, lang) => new NeuralNetworkSim(el, {
    lang: lang,
    dataset: attr(el, 'dataset', 'xor'),
    layers: attr(el, 'layers', 0),
    neurons: attr(el, 'neurons', 4),
    activation: attr(el, 'activation', 'tanh'),
    lr: attr(el, 'lr', 0.3),
    // Same seed and epochs give the same trained network (an embedded, already trained example).
    seed: attr(el, 'seed', 1),
    epochs: attr(el, 'epochs', 0)
});

const convolution = (el, lang) => new ConvolutionSim(el, {
    lang: lang,
    image: attr(el, 'image', 'shapes'),
    // A preset name, or the 9 weights as "a,b,c,...", with an optional divisor and absolute value.
    preset: attr(el, 'preset', 'identity'),
    kernel: attr(el, 'kernel', ''),
    divisor: attr(el, 'divisor', ''),
    abs: attr(el, 'abs', undefined),
    target: attr(el, 'target', ''),
    presets: attr(el, 'presets', true)
});

const qLearning = (el, lang) => new QLearningSim(el, {
    lang: lang,
    map: attr(el, 'map', 'maze'),
    alpha: attr(el, 'alpha', 0.5),
    gamma: attr(el, 'gamma', 0.9),
    epsilon: attr(el, 'epsilon', 0.2),
    // Same seed and episodes give the same Q-table (an embedded, already trained agent).
    seed: attr(el, 'seed', 1),
    episodes: attr(el, 'episodes', 0)
});

/** Simulation factories by data-aisim value (several aliases per simulation). */
const SIMS = {
    'gradient-descent': gradientDescent,
    'gradient_descent': gradientDescent,
    'confusion-matrix': confusionMatrix,
    'confusion_matrix': confusionMatrix,
    'regression': regression,
    'polynomial-regression': regression,
    'knn-tree': knnTree('knn'),
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
    'reinforcement-learning': qLearning
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
            SIMS[simType](el, document.documentElement.lang || 'en');
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
