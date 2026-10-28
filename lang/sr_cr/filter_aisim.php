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

// Опште команде.
$string['run'] = 'Покрени';
$string['pause'] = 'Пауза';
$string['step'] = 'Корак';
$string['reset'] = 'Ресетуј';
$string['speed'] = 'Брзина';
$string['status_ready'] = 'Спремно. Кликните на Покрени или Корак.';
$string['status_running'] = 'Обучавање у току...';

// Градијентни спуст.
$string['gradient_descent'] = 'Градијентни спуст';
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
$string['status_converged'] = 'Конвергирало ка минимуму!';
$string['status_diverging'] = 'Упозорење: Дивергенција! Стопа учења је превелика.';
$string['status_oscillating'] = 'Осцилује око минимума.';

// Матрица конфузије.
$string['confusion_matrix'] = 'Матрица конфузије и праг одлучивања';
$string['threshold'] = 'Праг одлучивања (θ)';
$string['actual_pos'] = 'Стварно позитивно (1)';
$string['actual_neg'] = 'Стварно негативно (0)';
$string['pred_pos'] = 'Предвиђено (1)';
$string['pred_neg'] = 'Предвиђено (0)';
$string['tp'] = 'Право позитивно (TP)';
$string['tn'] = 'Право негативно (TN)';
$string['fp'] = 'Лажно позитивно (FP - Грешка типа I)';
$string['fn'] = 'Лажно негативно (FN - Грешка типа II)';
$string['accuracy'] = 'Тачност (Accuracy)';
$string['precision'] = 'Прецизност (Precision)';
$string['recall'] = 'Опозив (Recall / Осетљивост)';
$string['f1_score'] = 'F1-скор';
$string['roc_curve'] = 'ROC крива (AUC)';
$string['scenario_quality'] = 'Индустријска контрола квалитета (Шкарт)';
$string['scenario_medical'] = 'Медицинска дијагностика (Скрининг)';
$string['scenario_spam'] = 'Класификација нежељене поште (Spam)';

// Регресија и генерализација.
$string['regression'] = 'Полиномска регресија и преприлагођавање';
$string['degree'] = 'Степен полинома (d)';
$string['regularization'] = 'Регуларизација (λ)';
$string['train_mse'] = 'Тренинг MSE грешка';
$string['val_mse'] = 'Валидациони MSE';
$string['underfitting'] = 'Потприлагођавање (Underfitting)';
$string['overfitting'] = 'Преприлагођавање (Overfitting)';
$string['good_fit'] = 'Оптимално уклапање (Добра генерализација)';

// k-NN и Стабло одлучивања.
$string['knn_tree'] = 'Границе одлучивања (k-NN и Стабло)';
$string['knn'] = 'k-Најближих суседа (k-NN)';
$string['decision_tree'] = 'Стабло одлучивања';
$string['k_neighbors'] = 'Број суседа (k)';
$string['tree_depth'] = 'Максимална дубина стабла';
$string['dataset_moons'] = 'Два полумесеца';
$string['dataset_circles'] = 'Концентрични кругови';
$string['dataset_blobs'] = 'Два кластера';
