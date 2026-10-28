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
 * Strings for component 'filter_aisim', language 'en'.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['filtername'] = 'AI Simulations Viewer';
$string['pluginname'] = 'AI Simulations Viewer';
$string['privacy:metadata'] = 'The AI Simulations Viewer filter does not store any personal data.';

// Gradient descent strings.
$string['gradient_descent'] = 'Gradient Descent';
$string['run'] = 'Run';
$string['pause'] = 'Pause';
$string['step'] = 'Step';
$string['reset'] = 'Reset';
$string['speed'] = 'Speed';
$string['learning_rate'] = 'Learning Rate (α)';
$string['initial_weight'] = 'Initial weight (w₀)';
$string['current_weight'] = 'Weight (w)';
$string['current_cost'] = 'Cost J(w)';
$string['current_gradient'] = 'Gradient dJ/dw';
$string['epoch'] = 'Epoch';
$string['cost_history'] = 'Cost History J(w)';
$string['profile_convex'] = 'Convex (Single Minimum)';
$string['profile_nonconvex'] = 'Non-convex (Local Minima)';
$string['profile_plateau'] = 'Saddle / Plateau';
$string['status_ready'] = 'Ready. Click Run or Step.';
$string['status_running'] = 'Training in progress...';
$string['status_converged'] = 'Converged to minimum!';
$string['status_diverging'] = 'Warning: Diverging! Learning rate is too high.';
$string['status_oscillating'] = 'Oscillating around minimum.';
