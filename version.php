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
 * AI simulation filter version details.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$plugin->component = 'filter_aisim';
$plugin->version   = 2026103003;
$plugin->requires  = 2024100700; // Moodle 4.5.
$plugin->supported = [405, 503];
$plugin->maturity  = MATURITY_ALPHA;
$plugin->release   = '0.9.0';
$plugin->source    = 'https://github.com/blagojevicboban/moodle-filter_aisim';
$plugin->tracker   = 'https://github.com/blagojevicboban/moodle-filter_aisim/issues';
$plugin->documentation = 'https://github.com/blagojevicboban/moodle-filter_aisim/blob/master/docs/PRIRUCNIK_ZA_NASTAVNIKE.md';
