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
 * Language strings for the simulation engines.
 *
 * The engines are plain classes shared with the standalone AISim Lab, so they do not call core/str themselves:
 * they get an object of their strings (identifier => text) in options.strings.
 *
 * @module filter_aisim/strings
 * @copyright 2026 Boban Blagojević
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
