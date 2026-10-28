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
 * Strings for component 'filter_aisim', language 'sr_lt'.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['filtername'] = 'Pregledač AI simulacija';
$string['pluginname'] = 'Pregledač AI simulacija';
$string['privacy:metadata'] = 'Filter za pregled AI simulacija ne čuva nikakve lične podatke.';

// Gradijentni spust.
$string['gradient_descent'] = 'Gradijentni spust';
$string['run'] = 'Pokreni';
$string['pause'] = 'Pauza';
$string['step'] = 'Korak';
$string['reset'] = 'Resetuj';
$string['speed'] = 'Brzina';
$string['learning_rate'] = 'Stopa učenja (α)';
$string['initial_weight'] = 'Početna težina (w₀)';
$string['current_weight'] = 'Težina (w)';
$string['current_cost'] = 'Trošak J(w)';
$string['current_gradient'] = 'Gradijent dJ/dw';
$string['epoch'] = 'Epoha';
$string['cost_history'] = 'Istorija troška J(w)';
$string['profile_convex'] = 'Konveksna (jedan minimum)';
$string['profile_nonconvex'] = 'Nekonveksna (lokalni minimumi)';
$string['profile_plateau'] = 'Plato / sedlasta tačka';
$string['status_ready'] = 'Spremno. Kliknite na Pokreni ili Korak.';
$string['status_running'] = 'Obučavanje u toku...';
$string['status_converged'] = 'Konvergiralo ka minimumu!';
$string['status_diverging'] = 'Upozorenje: Divergencija! Stopa učenja je prevelika.';
$string['status_oscillating'] = 'Osciluje oko minimuma.';
