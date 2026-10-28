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
 * Strings for component 'filter_aisim', language 'sr_cr'.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['filtername'] = 'Прегледач AI симулација';
$string['pluginname'] = 'Прегледач AI симулација';
$string['privacy:metadata'] = 'Филтер за преглед AI симулација не чува никакве личне податке.';

// Градијентни спуст.
$string['gradient_descent'] = 'Градијентни спуст';
$string['run'] = 'Покрени';
$string['pause'] = 'Пауза';
$string['step'] = 'Корак';
$string['reset'] = 'Ресетуј';
$string['speed'] = 'Брзина';
$string['learning_rate'] = 'Стопа учења (α)';
$string['initial_weight'] = 'Почетна тежина (w₀)';
$string['current_weight'] = 'Тежина (w)';
$string['current_cost'] = 'Трошак J(w)';
$string['current_gradient'] = 'Градијент dJ/dw';
$string['epoch'] = 'Епоха';
$string['cost_history'] = 'Историја трошка J(w)';
$string['profile_convex'] = 'Конвексна (један минимум)';
$string['profile_nonconvex'] = 'Неконвексна (локални минимуми)';
$string['profile_plateau'] = 'Плато / седласта тачка';
$string['status_ready'] = 'Спремно. Кликните на Покрени или Корак.';
$string['status_running'] = 'Обучавање у току...';
$string['status_converged'] = 'Конвергирало ка минимуму!';
$string['status_diverging'] = 'Упозорење: Дивергенција! Стопа учења је превелика.';
$string['status_oscillating'] = 'Осцилује око минимума.';
