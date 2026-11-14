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
 * Cross-Validation interactive simulation engine.
 *
 * The hyperparameter is the degree of a polynomial. The training data is split into k folds; every fold is held out
 * once and the model trained on the other folds is measured on it. The mean of the fold errors chooses the degree, and
 * a separate test set, locked until the end, tells how good the chosen model really is.
 *
 * @module filter_aisim/sims/cross_validation
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {renderTemplate, whenRendered, selectOptions} from 'filter_aisim/sims/layout';

const DATASETS = ['sine', 'sensor', 'noisy'];
const MAX_DEGREE = 9;
const N_TRAIN = 20;
const N_TEST = 12;
const FOLD_COLOURS = ['#2563eb', '#16a34a', '#d97706', '#9333ea', '#0891b2', '#db2777', '#65a30d', '#7c3aed', '#ea580c',
    '#0f766e'];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const round = (v, digits) => Math.round(v * Math.pow(10, digits)) / Math.pow(10, digits);

/**
 * Small seeded random generator (Park–Miller), returns numbers in (0, 1).
 *
 * @param {Number} seed
 * @returns {Function}
 */
const seeded = (seed) => {
    let s = (Math.abs(Math.floor(seed)) % 2147483646) + 1;
    return () => {
        s = (s * 16807) % 2147483647;
        return (s - 1) / 2147483646;
    };
};

/**
 * Least-squares polynomial of a degree through points {x, y} (a tiny ridge term keeps it solvable).
 *
 * @param {Array} pts
 * @param {Number} degree
 * @returns {Array} coefficients, lowest power first
 */
const fitPolynomial = (pts, degree) => {
    const m = degree + 1;
    const A = Array.from({length: m}, () => new Float64Array(m + 1));
    for (const {x, y} of pts) {
        const pw = [1];
        for (let p = 1; p <= 2 * degree; p++) {
            pw.push(pw[p - 1] * x);
        }
        for (let r = 0; r < m; r++) {
            for (let c = 0; c < m; c++) {
                A[r][c] += pw[r + c];
            }
            A[r][m] += pw[r] * y;
        }
    }
    for (let r = 1; r < m; r++) {
        A[r][r] += 1e-6;
    }
    for (let i = 0; i < m; i++) {
        let best = i;
        for (let k = i + 1; k < m; k++) {
            if (Math.abs(A[k][i]) > Math.abs(A[best][i])) {
                best = k;
            }
        }
        [A[i], A[best]] = [A[best], A[i]];
        if (Math.abs(A[i][i]) < 1e-12) {
            continue;
        }
        for (let k = i + 1; k < m; k++) {
            const f = A[k][i] / A[i][i];
            for (let j = i; j <= m; j++) {
                A[k][j] -= f * A[i][j];
            }
        }
    }
    const w = new Array(m).fill(0);
    for (let i = m - 1; i >= 0; i--) {
        let sum = A[i][m];
        for (let j = i + 1; j < m; j++) {
            sum -= A[i][j] * w[j];
        }
        w[i] = Math.abs(A[i][i]) < 1e-12 ? 0 : sum / A[i][i];
    }
    return w;
};

const evalPoly = (w, x) => w.reduceRight((acc, c) => acc * x + c, 0);
const mse = (w, pts) => pts.reduce((s, p) => s + (evalPoly(w, p.x) - p.y) ** 2, 0) / (pts.length || 1);

export class CrossValidationSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('cv_title'),
            badge: this.str('cv_badge'),
            dataTitle: this.str('cv_data_title'),
            trainPts: this.str('cv_train_pts'),
            testPts: this.str('cv_test_pts'),
            settings: this.str('cv_settings'),
            degree: this.str('cv_degree'),
            degreeHint: this.str('cv_degree_hint'),
            folds: this.str('cv_folds'),
            foldsHint: this.str('cv_folds_hint'),
            reshuffle: this.str('cv_reshuffle'),
            testBtn: this.str('cv_test_btn'),
            foldsTitle: this.str('cv_folds_title'),
            curveTitle: this.str('cv_curve_title'),
            trainShort: this.str('cv_train_short'),
            cvShort: this.str('cv_cv_short'),
            metrics: this.str('cv_metrics'),
            trainError: this.str('cv_train_error'),
            cvError: this.str('cv_cv_error'),
            cvSpread: this.str('cv_cv_spread'),
            bestDegree: this.str('cv_best_degree'),
            testError: this.str('cv_test_error'),
            testHidden: this.str('cv_test_hidden'),
            adviceStart: this.str('cv_advice_start'),
        };

        this.datasetKey = DATASETS.includes(options.dataset) ? options.dataset : 'sine';
        this.k = clamp(parseInt(options.k, 10) || 5, 2, 10);
        this.degree = clamp(parseInt(options.degree, 10) || 1, 1, MAX_DEGREE);
        this.seed = Math.max(1, parseInt(options.seed, 10) || 1);
        this.tested = options.tested === true || options.tested === 1 || options.tested === '1' || options.tested === 'true';
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.initData();
        this.assignFolds();
        whenRendered(this.renderLayout(), () => {
            this.bindEvents();
            this.update();
        });
    }

    /**
     * A UI text from the filter_aisim language strings, with {$a} or {$a->name} filled in.
     *
     * @param {String} key String identifier
     * @param {*} [a] Value, or object of named values, for the placeholders
     * @returns {String}
     */
    str(key, a) {
        const text = this.strings[key];
        if (text === undefined) {
            return '[[' + key + ']]';
        }
        return text.replace(/\{\$a(?:->(\w+))?\}/g, (m, name) => String(name ? a[name] : a));
    }

    /**
     * The measurements of the dataset (always the same for a dataset): N_TRAIN training points and a test set.
     */
    initData() {
        const rand = seeded(3000 + DATASETS.indexOf(this.datasetKey));
        const gauss = () => {
            const u = Math.max(rand(), 1e-9);
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
        };
        const f = {
            sine: (x) => Math.sin(Math.PI * x),
            sensor: (x) => 1.6 * (1 - Math.exp(-2.2 * (x + 1))) - 0.8,
            noisy: (x) => 0.9 * Math.sin(Math.PI * x),
        }[this.datasetKey];
        const noise = this.datasetKey === 'noisy' ? 0.7 : 0.3;
        const point = () => {
            const x = rand() * 2 - 1;
            return {x, y: f(x) + gauss() * noise};
        };
        this.train = Array.from({length: N_TRAIN}, point);
        this.test = Array.from({length: N_TEST}, point);
        this.truth = f;
    }

    /**
     * Shuffle the training points (seeded) and deal them into k folds.
     */
    assignFolds() {
        const rand = seeded(this.seed * 7907 + 13);
        const order = this.train.map((p, i) => i);
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }
        this.fold = new Array(this.train.length);
        order.forEach((idx, pos) => {
            this.fold[idx] = pos % this.k;
        });
        this.computeCurve();
    }

    /**
     * Validation error of every fold for a degree.
     *
     * @param {Number} degree
     * @returns {Array}
     */
    foldErrors(degree) {
        const errors = [];
        for (let f = 0; f < this.k; f++) {
            const fit = this.train.filter((p, i) => this.fold[i] !== f);
            const val = this.train.filter((p, i) => this.fold[i] === f);
            errors.push(mse(fitPolynomial(fit, degree), val));
        }
        return errors;
    }

    /**
     * Training and cross-validation error for every degree.
     */
    computeCurve() {
        this.curve = [];
        for (let d = 1; d <= MAX_DEGREE; d++) {
            const errs = this.foldErrors(d);
            const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
            const sd = Math.sqrt(errs.reduce((a, b) => a + (b - mean) ** 2, 0) / errs.length);
            const train = mse(fitPolynomial(this.train, d), this.train);
            this.curve.push({d, errs, mean, sd, train, min: Math.min(...errs), max: Math.max(...errs)});
        }
        this.best = this.curve.reduce((b, c) => (c.mean < b.mean ? c : b)).d;
    }

    async renderLayout() {
        const context = {
            t: this.t,
            datasets: selectOptions([
                ['sine', this.str('cv_datasets_sine')],
                ['sensor', this.str('cv_datasets_sensor')],
                ['noisy', this.str('cv_datasets_noisy')],
            ], this.datasetKey),
            degree: {id: 'aisim-cv-degree', label: this.t.degree, value: this.degree, text: String(this.degree),
                min: 1, max: MAX_DEGREE, step: 1},
            folds: {id: 'aisim-cv-k', label: this.t.folds, value: this.k, text: String(this.k), min: 2, max: 10, step: 1},
        };
        await renderTemplate(this.container, 'cross_validation', context);
        this.canvas = this.container.querySelector('#aisim-cv-canvas');
        this.foldCanvas = this.container.querySelector('#aisim-cv-folds');
        this.curveCanvas = this.container.querySelector('#aisim-cv-curve');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv, fallbackH] of [['main', this.canvas, 280], ['folds', this.foldCanvas, 150],
                ['curve', this.curveCanvas, 170]]) {
            const rect = cv.getBoundingClientRect();
            const w = rect.width || 400;
            const h = rect.height || fallbackH;
            cv.width = w * dpr;
            cv.height = h * dpr;
            const ctx = cv.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this.sizesPx[key] = {w, h, ctx};
        }
    }

    bindEvents() {
        const q = (id) => this.container.querySelector(id);
        const controls = ['#aisim-cv-dataset', '#aisim-cv-degree', '#aisim-cv-k', '#aisim-cv-reshuffle', '#aisim-cv-test'];
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        q('#aisim-cv-dataset').addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.tested = false;
            this.initData();
            this.assignFolds();
            this.update();
        });
        q('#aisim-cv-degree').addEventListener('input', (e) => {
            this.degree = parseInt(e.target.value, 10);
            this.update();
        });
        q('#aisim-cv-k').addEventListener('input', (e) => {
            this.k = parseInt(e.target.value, 10);
            this.assignFolds();
            this.update();
        });
        q('#aisim-cv-reshuffle').addEventListener('click', () => {
            this.seed++;
            this.assignFolds();
            this.update();
        });
        q('#aisim-cv-test').addEventListener('click', () => {
            this.tested = true;
            this.update();
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
    }

    update() {
        const q = (id) => this.container.querySelector(id);
        const cur = this.curve[this.degree - 1];
        const model = fitPolynomial(this.train, this.degree);
        const testMse = mse(model, this.test);
        q('#aisim-cv-degree-val').textContent = this.degree;
        q('#aisim-cv-k-val').textContent = this.k;
        q('#aisim-cv-train').textContent = cur.train.toFixed(3);
        q('#aisim-cv-mean').textContent = cur.mean.toFixed(3);
        q('#aisim-cv-sd').textContent = '± ' + cur.sd.toFixed(3);
        q('#aisim-cv-best').textContent = 'd = ' + this.best;
        q('#aisim-cv-testval').textContent = this.tested ? testMse.toFixed(3) : this.t.testHidden;
        q('#aisim-cv-advice').textContent = this.advice(cur);
        this.model = model;
        this.draw();

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'cross_validation',
                parameters: {dataset: this.datasetKey, k: this.k, degree: this.degree, seed: this.seed, tested: this.tested},
                metrics: {
                    cvMse: round(cur.mean, 4),
                    cvSd: round(cur.sd, 4),
                    trainMse: round(cur.train, 4),
                    bestDegree: this.best,
                    testMse: this.tested ? round(testMse, 4) : null
                }
            });
        }
    }

    advice(cur) {
        if (this.tested) {
            return this.str('cv_advice_tested');
        }
        if (this.k === 2) {
            return this.str('cv_advice_fewfolds');
        }
        if (this.degree === this.best) {
            return this.str('cv_advice_best');
        }
        if (this.degree < this.best) {
            return this.str('cv_advice_under');
        }
        if (cur.mean > this.curve[this.best - 1].mean * 1.3) {
            return this.str('cv_advice_over');
        }
        return this.t.adviceStart;
    }

    draw() {
        this.drawData();
        this.drawFolds();
        this.drawCurve();
    }

    drawData() {
        const {w, h, ctx} = this.sizesPx.main;
        ctx.clearRect(0, 0, w, h);
        const ys = this.train.concat(this.test).map((p) => p.y);
        const ymin = Math.min(...ys) - 0.3;
        const ymax = Math.max(...ys) + 0.3;
        const pad = 14;
        const toPx = (x, y) => ({
            px: pad + ((x + 1) / 2) * (w - 2 * pad),
            py: h - pad - ((clamp(y, ymin - 1, ymax + 1) - ymin) / (ymax - ymin)) * (h - 2 * pad),
        });

        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        const z = toPx(-1, 0);
        ctx.beginPath();
        ctx.moveTo(pad, z.py);
        ctx.lineTo(w - pad, z.py);
        ctx.stroke();

        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.clip();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i <= 160; i++) {
            const x = -1 + (2 * i) / 160;
            const {px, py} = toPx(x, evalPoly(this.model, x));
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();
        ctx.restore();

        this.train.forEach((p, i) => {
            const {px, py} = toPx(p.x, p.y);
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            ctx.fillStyle = FOLD_COLOURS[this.fold[i] % FOLD_COLOURS.length];
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.stroke();
        });
        for (const p of this.test) {
            const {px, py} = toPx(p.x, p.y);
            ctx.lineWidth = 1.8;
            ctx.strokeStyle = this.tested ? '#ef4444' : '#94a3b8';
            ctx.strokeRect(px - 4, py - 4, 8, 8);
        }
    }

    drawFolds() {
        const {w, h, ctx} = this.sizesPx.folds;
        ctx.clearRect(0, 0, w, h);
        const cur = this.curve[this.degree - 1];
        const top = Math.max(...cur.errs, cur.mean) * 1.15 || 1;
        const pad = 18;
        const bw = (w - 2 * pad) / this.k;
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        cur.errs.forEach((e, f) => {
            const bh = (Math.min(e, top) / top) * (h - 2 * pad);
            const x = pad + f * bw + bw * 0.15;
            ctx.fillStyle = FOLD_COLOURS[f % FOLD_COLOURS.length];
            ctx.fillRect(x, h - pad - bh, bw * 0.7, bh);
            ctx.fillStyle = '#475569';
            ctx.fillText(String(f + 1), x + bw * 0.35, h - 4);
        });
        const my = h - pad - (cur.mean / top) * (h - 2 * pad);
        ctx.strokeStyle = '#0f172a';
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pad, my);
        ctx.lineTo(w - pad, my);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.textAlign = 'left';
        ctx.fillStyle = '#0f172a';
        ctx.fillText('MSE ' + cur.mean.toFixed(3), pad, Math.max(11, my - 4));
    }

    drawCurve() {
        const {w, h, ctx} = this.sizesPx.curve;
        ctx.clearRect(0, 0, w, h);
        const pad = 18;
        // Cap the axis so that a wildly overfitted degree does not flatten the rest of the curve.
        const best = this.curve[this.best - 1].mean;
        const top = Math.min(Math.max(...this.curve.map((c) => c.max)), best * 6) * 1.05;
        const toPx = (d, v) => ({
            px: pad + ((d - 1) / (MAX_DEGREE - 1)) * (w - 2 * pad),
            py: h - pad - (Math.min(v, top) / top) * (h - 2 * pad),
        });

        // Band from the best to the worst fold.
        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        ctx.beginPath();
        this.curve.forEach((c, i) => {
            const {px, py} = toPx(c.d, c.max);
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        });
        [...this.curve].reverse().forEach((c) => {
            const {px, py} = toPx(c.d, c.min);
            ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fill();

        for (const [key, colour] of [['train', '#2563eb'], ['mean', '#ef4444']]) {
            ctx.strokeStyle = colour;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            this.curve.forEach((c, i) => {
                const {px, py} = toPx(c.d, c[key]);
                if (i === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            });
            ctx.stroke();
        }

        const b = toPx(this.best, best);
        ctx.fillStyle = '#f59e0b';
        ctx.font = '16px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('★', b.px, b.py - 4);

        const cx = toPx(this.degree, 0).px;
        ctx.strokeStyle = '#0f172a';
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, pad);
        ctx.lineTo(cx, h - pad);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = '11px system-ui, sans-serif';
        ctx.fillStyle = '#475569';
        for (let d = 1; d <= MAX_DEGREE; d++) {
            ctx.fillText(String(d), toPx(d, 0).px, h - 4);
        }
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
CrossValidationSim.STRINGS = [
    'cv_title', 'cv_badge', 'cv_data_title', 'cv_train_pts', 'cv_test_pts', 'cv_settings', 'cv_degree',
    'cv_degree_hint', 'cv_folds', 'cv_folds_hint', 'cv_reshuffle', 'cv_test_btn', 'cv_folds_title', 'cv_curve_title',
    'cv_train_short', 'cv_cv_short', 'cv_metrics', 'cv_train_error', 'cv_cv_error', 'cv_cv_spread', 'cv_best_degree',
    'cv_test_error', 'cv_test_hidden', 'cv_advice_start', 'cv_advice_tested', 'cv_advice_fewfolds', 'cv_advice_best',
    'cv_advice_under', 'cv_advice_over', 'cv_datasets_sine', 'cv_datasets_sensor', 'cv_datasets_noisy'
];
