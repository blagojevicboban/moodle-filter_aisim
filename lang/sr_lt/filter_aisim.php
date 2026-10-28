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

// Opšte komande.
$string['run'] = 'Pokreni';
$string['pause'] = 'Pauza';
$string['step'] = 'Korak';
$string['reset'] = 'Resetuj';
$string['speed'] = 'Brzina';
$string['status_ready'] = 'Spremno. Kliknite na Pokreni ili Korak.';
$string['status_running'] = 'Obučavanje u toku...';

// Gradijentni spust.
$string['gradient_descent'] = 'Gradijentni spust';
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
$string['status_converged'] = 'Konvergiralo ka minimumu!';
$string['status_diverging'] = 'Upozorenje: Divergencija! Stopa učenja je prevelika.';
$string['status_oscillating'] = 'Osciluje oko minimuma.';

// Matrica konfuzije.
$string['confusion_matrix'] = 'Matrica konfuzije i prag odlučivanja';
$string['threshold'] = 'Prag odlučivanja (θ)';
$string['actual_pos'] = 'Stvarno pozitivno (1)';
$string['actual_neg'] = 'Stvarno negativno (0)';
$string['pred_pos'] = 'Predviđeno (1)';
$string['pred_neg'] = 'Predviđeno (0)';
$string['tp'] = 'Pravo pozitivno (TP)';
$string['tn'] = 'Pravo negativno (TN)';
$string['fp'] = 'Lažno pozitivno (FP - Greška tipa I)';
$string['fn'] = 'Lažno negativno (FN - Greška tipa II)';
$string['accuracy'] = 'Tačnost (Accuracy)';
$string['precision'] = 'Preciznost (Precision)';
$string['recall'] = 'Opoziv (Recall / Osetljivost)';
$string['f1_score'] = 'F1-skor';
$string['roc_curve'] = 'ROC kriva (AUC)';
$string['scenario_quality'] = 'Industrijska kontrola kvaliteta (Škart)';
$string['scenario_medical'] = 'Medicinska dijagnostika (Skrining)';
$string['scenario_spam'] = 'Klasifikacija neželjene pošte (Spam)';

// Regresija i generalizacija.
$string['regression'] = 'Polinomska regresija i preprilagođavanje';
$string['degree'] = 'Stepen polinoma (d)';
$string['regularization'] = 'Regularizacija (λ)';
$string['train_mse'] = 'Trening MSE greška';
$string['val_mse'] = 'Validacioni MSE';
$string['underfitting'] = 'Potprilagođavanje (Underfitting)';
$string['overfitting'] = 'Preprilagođavanje (Overfitting)';
$string['good_fit'] = 'Optimalno uklapanje (Dobra generalizacija)';
