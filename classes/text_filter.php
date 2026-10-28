<?php
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
 * AI simulation viewer filter.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace filter_aisim;

/**
 * Renders interactive AI simulations (data-aisim) into live interactive widgets.
 *
 * @package    filter_aisim
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class text_filter extends \core_filters\text_filter {
    /**
     * Load the viewer AMD module when text contains an interactive AI simulation.
     *
     * @param string $text
     * @param array $options
     * @return string
     */
    public function filter($text, array $options = []) {
        global $PAGE;
        if (!is_string($text) || strpos($text, 'data-aisim') === false) {
            return $text;
        }
        $PAGE->requires->js_call_amd('filter_aisim/viewer', 'init');
        return $text;
    }
}
