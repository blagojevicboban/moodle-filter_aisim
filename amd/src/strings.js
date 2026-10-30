/**
 * Language strings for the simulation engines.
 *
 * The engines are plain classes shared with the standalone AISim Lab, so they do not call core/str themselves:
 * they get an object of their strings (identifier => text) in options.strings.
 *
 * @module filter_aisim/strings
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {getStrings} from 'core/str';

/**
 * Load filter_aisim strings in the current language.
 *
 * @param {String[]} keys String identifiers, e.g. an engine's STRINGS list
 * @returns {Promise<Object>} identifier => text, placeholders such as {$a} left for the engine to fill in
 */
export const load = (keys) => getStrings(keys.map((key) => ({key, component: 'filter_aisim'})))
    .then((values) => Object.fromEntries(keys.map((key, i) => [key, values[i]])));
