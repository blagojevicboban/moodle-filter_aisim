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
 * Shared helpers for the simulation layouts, which live in Mustache templates.
 *
 * @module filter_aisim/sims/layout
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import Templates from 'core/templates';
import Notification from 'core/notification';

/**
 * Render a simulation layout template into its container, replacing what was there.
 *
 * @param {HTMLElement} container Simulation root element
 * @param {String} name Template name inside filter_aisim
 * @param {Object} context Template context
 * @returns {Promise}
 */
export const renderTemplate = (container, name, context) =>
    Templates.renderForPromise('filter_aisim/' + name, context)
        .then(({html, js}) => Templates.replaceNodeContents(container, html, js));

/**
 * Wait for a layout render, then run the code that needs the new elements; errors go to the Moodle notification.
 *
 * @param {Promise} rendered Promise from a renderLayout() call
 * @param {Function} then Code that binds events and draws
 * @returns {Promise}
 */
export const whenRendered = (rendered, then) => rendered.then(then).catch(Notification.exception);

/**
 * Fill a calculation box with a bold heading and one line per step (plain text, no HTML).
 *
 * @param {HTMLElement} box Element to fill
 * @param {String} heading First line, in bold
 * @param {Array} steps One entry per line: a string, or an array of parts where a string is text and {span: text}
 *     is a <span>
 */
export const showSteps = (box, heading, steps) => {
    const head = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = heading;
    head.append(strong);
    const lines = steps.map((step) => {
        const line = document.createElement('div');
        line.className = 'aisim-conv-step';
        line.append(...[].concat(step).map((part) => {
            if (typeof part === 'string') {
                return part;
            }
            const span = document.createElement('span');
            span.textContent = part.span;
            return span;
        }));
        return line;
    });
    box.replaceChildren(head, ...lines);
};

/**
 * Options of a select element for a template: [{value, label, selected}].
 *
 * @param {Array} pairs [[value, label], ...]
 * @param {String} current Selected value
 * @returns {Array}
 */
export const selectOptions = (pairs, current) =>
    pairs.map(([value, label]) => ({value, label, selected: String(value) === String(current)}));
