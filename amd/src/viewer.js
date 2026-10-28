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
import {eventTypes} from 'core_filters/events';

const SELECTOR = '[data-aisim]';
let initialised = false;

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
        const lang = document.documentElement.lang || 'en';

        if (simType === 'gradient-descent' || simType === 'gradient_descent') {
            new GradientDescentSim(el, {
                lang: lang,
                profile: el.getAttribute('data-profile') || 'convex',
                lr: el.getAttribute('data-lr') || '',
                w0: el.getAttribute('data-w0') || ''
            });
        } else if (simType === 'confusion-matrix' || simType === 'confusion_matrix') {
            new ConfusionMatrixSim(el, {
                lang: lang,
                scenario: el.getAttribute('data-scenario') || 'quality',
                threshold: el.getAttribute('data-threshold') || 0.50
            });
        } else if (simType === 'regression' || simType === 'polynomial-regression') {
            new RegressionSim(el, {
                lang: lang,
                dataset: el.getAttribute('data-dataset') || 'sine',
                degree: el.getAttribute('data-degree') || 3,
                lambda: el.getAttribute('data-lambda') || 0.0
            });
        } else if (simType === 'knn-tree' || simType === 'knn' || simType === 'decision-tree') {
            new KnnTreeSim(el, {
                lang: lang,
                model: simType === 'decision-tree' ? 'tree' : 'knn',
                dataset: el.getAttribute('data-dataset') || 'moons',
                k: el.getAttribute('data-k') || 3,
                depth: el.getAttribute('data-depth') || 3
            });
        } else if (simType === 'kmeans' || simType === 'k-means') {
            new KMeansSim(el, {
                lang: lang,
                dataset: el.getAttribute('data-dataset') || 'blobs3',
                k: el.getAttribute('data-k') || 3
            });
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
