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
        const opts = {
            lang: document.documentElement.lang || 'en',
            profile: el.getAttribute('data-profile') || 'convex',
            lr: el.getAttribute('data-lr') || '',
            w0: el.getAttribute('data-w0') || ''
        };

        if (simType === 'gradient-descent' || simType === 'gradient_descent') {
            new GradientDescentSim(el, opts);
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
