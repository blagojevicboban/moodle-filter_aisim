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
 * Logistic Regression interactive simulation engine.
 *
 * Two input variables, z = w1·x1 + w2·x2 + b, probability σ(z) and a decision threshold θ. The weights are set by hand
 * or learned with gradient descent on the log-loss; a click on a point shows its calculation.
 *
 * @module filter_aisim/sims/logistic
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {renderTemplate, whenRendered, selectOptions, showSteps} from 'filter_aisim/sims/layout';

const DATASETS = ['machine', 'exam', 'ring'];
const MAX_EPOCHS = 2000;
const WMAX = 8;
/** The plot shows x1 and x2 from -RANGE to RANGE. */
const RANGE = 2.2;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
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

export class LogisticSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('log_title'),
            badge: this.str('log_badge'),
            model: this.str('log_model'),
            w1: this.str('log_w1'),
            w2: this.str('log_w2'),
            bias: this.str('log_bias'),
            threshold: this.str('log_threshold'),
            thresholdHint: this.str('log_threshold_hint'),
            training: this.str('log_training'),
            lr: this.str('log_lr'),
            train: this.str('log_train'),
            pause: this.str('log_pause'),
            step: this.str('log_step'),
            reset: this.str('log_reset'),
            metrics: this.str('log_metrics'),
            epoch: this.str('log_epoch'),
            loss: this.str('log_loss'),
            accuracy: this.str('log_accuracy'),
            precision: this.str('log_precision'),
            recall: this.str('log_recall'),
            lossCurve: this.str('log_loss_curve'),
            mapTitle: this.str('log_map_title'),
            sigmoidTitle: this.str('log_sigmoid_title'),
            boundary: this.str('log_boundary'),
            calc: this.str('log_calc'),
            calcHint: this.str('log_calc_hint'),
            adviceStart: this.str('log_advice_start'),
            adviceGood: this.str('log_advice_good'),
            adviceThreshold: this.str('log_advice_threshold'),
            adviceRing: this.str('log_advice_ring'),
        };

        this.datasetKey = DATASETS.includes(options.dataset) ? options.dataset : 'machine';
        this.w1 = this.num(options.w1, 0, -WMAX, WMAX);
        this.w2 = this.num(options.w2, 0, -WMAX, WMAX);
        this.b = this.num(options.b, 0, -WMAX, WMAX);
        this.threshold = this.num(options.threshold, 0.5, 0.05, 0.95);
        this.lr = this.num(options.lr, 0.5, 0.05, 3);
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;
        this.running = false;
        this.lastNotify = 0;
        this.selected = null;

        this.initData();
        this.resetTraining();
        whenRendered(this.renderLayout(), () => {
            this.bindEvents();
            // A saved state (quiz review, shared links) restores the weights; the epochs are only shown again.
            this.epoch = Math.min(MAX_EPOCHS, Math.max(0, parseInt(options.epochs, 10) || 0));
            this.update(true);
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
     * A number option within limits, or the default when it is missing.
     *
     * @param {*} value
     * @param {Number} fallback
     * @param {Number} lo
     * @param {Number} hi
     * @returns {Number}
     */
    num(value, fallback, lo, hi) {
        const v = parseFloat(value);
        return isFinite(v) ? clamp(v, lo, hi) : fallback;
    }

    /**
     * Names of the axes and of the two classes of the current dataset.
     *
     * @returns {Object}
     */
    labels() {
        switch (this.datasetKey) {
            case 'exam':
                return {x1: this.str('log_exam_x1'), x2: this.str('log_exam_x2'),
                    cls1: this.str('log_exam_cls1'), cls0: this.str('log_exam_cls0')};
            case 'ring':
                return {x1: this.str('log_ring_x1'), x2: this.str('log_ring_x2'),
                    cls1: this.str('log_ring_cls1'), cls0: this.str('log_ring_cls0')};
            default:
                return {x1: this.str('log_machine_x1'), x2: this.str('log_machine_x2'),
                    cls1: this.str('log_machine_cls1'), cls0: this.str('log_machine_cls0')};
        }
    }

    initData() {
        const rand = seeded(2000 + DATASETS.indexOf(this.datasetKey));
        const gauss = () => {
            const u = Math.max(rand(), 1e-9);
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
        };
        const cloud = (n, cx, cy, sd, cls) => {
            for (let i = 0; i < n; i++) {
                this.points.push({x1: clamp(cx + gauss() * sd, -2, 2), x2: clamp(cy + gauss() * sd, -2, 2), cls});
            }
        };
        this.points = [];
        if (this.datasetKey === 'machine') {
            cloud(26, -0.6, -0.4, 0.6, 0);
            cloud(24, 0.6, 0.5, 0.6, 1);
        } else if (this.datasetKey === 'exam') {
            cloud(24, -0.6, -0.5, 0.75, 0);
            cloud(26, 0.5, 0.4, 0.75, 1);
        } else {
            for (let i = 0; i < 22; i++) {
                const a = 2 * Math.PI * rand();
                const r = 0.8 * Math.sqrt(rand());
                this.points.push({x1: Math.cos(a) * r, x2: Math.sin(a) * r, cls: 1});
            }
            for (let i = 0; i < 28; i++) {
                const a = 2 * Math.PI * rand();
                const r = 1.3 + rand() * 0.6;
                this.points.push({x1: Math.cos(a) * r, x2: Math.sin(a) * r, cls: 0});
            }
        }
    }

    resetTraining() {
        this.epoch = 0;
        this.history = [];
    }

    z(pt) {
        return this.w1 * pt.x1 + this.w2 * pt.x2 + this.b;
    }

    /**
     * One epoch of full-batch gradient descent on the log-loss: ∂L/∂w = mean((σ(z) − y)·x).
     */
    trainEpoch() {
        let g1 = 0;
        let g2 = 0;
        let gb = 0;
        for (const pt of this.points) {
            const err = sigmoid(this.z(pt)) - pt.cls;
            g1 += err * pt.x1;
            g2 += err * pt.x2;
            gb += err;
        }
        const n = this.points.length;
        this.w1 = clamp(this.w1 - this.lr * g1 / n, -WMAX, WMAX);
        this.w2 = clamp(this.w2 - this.lr * g2 / n, -WMAX, WMAX);
        this.b = clamp(this.b - this.lr * gb / n, -WMAX, WMAX);
        this.epoch++;
    }

    trainEpochs(count) {
        for (let i = 0; i < count && this.epoch < MAX_EPOCHS; i++) {
            this.trainEpoch();
            if (this.epoch % 5 === 0) {
                this.history.push(this.evaluate().loss);
            }
        }
    }

    /**
     * Log-loss, and accuracy, precision and recall of class 1 at the threshold θ.
     *
     * @returns {Object}
     */
    evaluate() {
        let loss = 0;
        let tp = 0;
        let fp = 0;
        let fn = 0;
        let correct = 0;
        for (const pt of this.points) {
            const p = clamp(sigmoid(this.z(pt)), 1e-7, 1 - 1e-7);
            loss -= pt.cls === 1 ? Math.log(p) : Math.log(1 - p);
            const pred = p >= this.threshold ? 1 : 0;
            correct += pred === pt.cls ? 1 : 0;
            tp += pred === 1 && pt.cls === 1 ? 1 : 0;
            fp += pred === 1 && pt.cls === 0 ? 1 : 0;
            fn += pred === 0 && pt.cls === 1 ? 1 : 0;
        }
        const n = this.points.length || 1;
        return {
            loss: loss / n,
            accuracy: correct / n,
            precision: tp + fp ? tp / (tp + fp) : 0,
            recall: tp + fn ? tp / (tp + fn) : 0,
        };
    }

    async renderLayout() {
        const l = this.labels();
        const context = {
            t: this.t,
            datasets: selectOptions([
                ['machine', this.str('log_datasets_machine')],
                ['exam', this.str('log_datasets_exam')],
                ['ring', this.str('log_datasets_ring')],
            ], this.datasetKey),
            cls1: l.cls1,
            cls0: l.cls0,
            sliders: [
                ['w1', this.t.w1 + ' w₁', this.w1, -WMAX, WMAX, 0.1, 2],
                ['w2', this.t.w2 + ' w₂', this.w2, -WMAX, WMAX, 0.1, 2],
                ['b', this.t.bias + ' b', this.b, -WMAX, WMAX, 0.1, 2],
            ].map(([key, label, value, min, max, step, digits]) => ({
                id: 'aisim-log-' + key, label, value, text: value.toFixed(digits), min, max, step,
            })),
            threshold: {
                id: 'aisim-log-th', label: 'θ', value: this.threshold, text: this.threshold.toFixed(2),
                min: 0.05, max: 0.95, step: 0.05,
            },
            lr: {
                id: 'aisim-log-lr', label: this.t.lr + ' α', value: this.lr, text: this.lr.toFixed(2),
                min: 0.05, max: 3, step: 0.05,
            },
        };
        await renderTemplate(this.container, 'logistic', context);
        this.canvas = this.container.querySelector('#aisim-log-canvas');
        this.sigCanvas = this.container.querySelector('#aisim-log-sigmoid');
        this.curve = this.container.querySelector('#aisim-log-curve');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv, fallbackH] of [['main', this.canvas, 320], ['sig', this.sigCanvas, 170],
                ['curve', this.curve, 110]]) {
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
        const controls = ['#aisim-log-dataset', '#aisim-log-w1', '#aisim-log-w2', '#aisim-log-b', '#aisim-log-th',
            '#aisim-log-lr', '#aisim-log-run', '#aisim-log-step', '#aisim-log-reset'];
        // The calculation of a point is shown in read-only mode too.
        this.canvas.addEventListener('click', (e) => this.selectAt(e));
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }

        q('#aisim-log-dataset').addEventListener('change', (e) => {
            this.stop();
            this.datasetKey = e.target.value;
            this.initData();
            this.resetTraining();
            this.selected = null;
            whenRendered(this.renderLayout(), () => {
                this.bindEvents();
                this.update(true);
            });
        });
        for (const key of ['w1', 'w2', 'b']) {
            q('#aisim-log-' + key).addEventListener('input', (e) => {
                this.stop();
                this[key] = parseFloat(e.target.value);
                this.update(true);
            });
        }
        q('#aisim-log-th').addEventListener('input', (e) => {
            this.threshold = parseFloat(e.target.value);
            this.update(true);
        });
        q('#aisim-log-lr').addEventListener('input', (e) => {
            this.lr = parseFloat(e.target.value);
            q('#aisim-log-lr-val').textContent = this.lr.toFixed(2);
        });
        q('#aisim-log-run').addEventListener('click', () => (this.running ? this.stop() : this.run()));
        q('#aisim-log-step').addEventListener('click', () => {
            this.stop();
            this.trainEpochs(10);
            this.update(true);
        });
        q('#aisim-log-reset').addEventListener('click', () => {
            this.stop();
            this.w1 = 0;
            this.w2 = 0;
            this.b = 0;
            this.resetTraining();
            this.update(true);
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
    }

    run() {
        this.running = true;
        this.container.querySelector('#aisim-log-run').textContent = this.t.pause;
        const frame = () => {
            if (!this.running) {
                return;
            }
            this.trainEpochs(5);
            this.update(false);
            if (this.epoch >= MAX_EPOCHS) {
                this.stop();
                return;
            }
            window.requestAnimationFrame(frame);
        };
        window.requestAnimationFrame(frame);
    }

    stop() {
        if (!this.running) {
            return;
        }
        this.running = false;
        this.container.querySelector('#aisim-log-run').textContent = this.t.train;
        this.update(true);
    }

    /**
     * Select the data point nearest to a click and show its calculation.
     *
     * @param {MouseEvent} e
     */
    selectAt(e) {
        const rect = this.canvas.getBoundingClientRect();
        const {w, h} = this.sizesPx.main;
        const x1 = -RANGE + ((e.clientX - rect.left) / w) * 2 * RANGE;
        const x2 = RANGE - ((e.clientY - rect.top) / h) * 2 * RANGE;
        let best = null;
        let bestD = 0.25 * 0.25;
        this.points.forEach((pt, i) => {
            const d = (pt.x1 - x1) ** 2 + (pt.x2 - x2) ** 2;
            if (d < bestD) {
                bestD = d;
                best = i;
            }
        });
        this.selected = best;
        this.showCalc();
        this.draw();
    }

    showCalc() {
        const box = this.container.querySelector('#aisim-log-calc');
        if (!box || this.selected === null) {
            return;
        }
        const pt = this.points[this.selected];
        const l = this.labels();
        const f = (v) => v.toFixed(2);
        const z = this.z(pt);
        const p = sigmoid(z);
        const pred = p >= this.threshold ? 1 : 0;
        const name = (cls) => (cls === 1 ? l.cls1 : l.cls0);
        const signed = (v) => (v < 0 ? ` − ${f(-v)}` : ` + ${f(v)}`);
        const factor = (v) => (v < 0 ? `(${f(v)})` : f(v));
        showSteps(box, this.str('log_calc_point', {x1: f(pt.x1), x2: f(pt.x2)}), [
            `z = ${f(this.w1)}·${factor(pt.x1)}${signed(this.w2)}·${factor(pt.x2)}${signed(this.b)} = ${f(z)}`,
            `σ(z) = 1 / (1 + e^(${f(-z)})) = ${p.toFixed(3)}`,
            [`${p.toFixed(3)} ${pred ? '≥' : '<'} θ = ${this.threshold.toFixed(2)} → ${this.str('log_calc_pred')} `,
                {span: name(pred)}],
            [this.str('log_calc_true') + ' ', {span: name(pt.cls) + (pred === pt.cls ? ' ✓' : ' ✗')}],
        ]);
    }

    /**
     * Refresh numbers and drawings; tell the listener (quiz answer) at most every 250 ms while training.
     *
     * @param {Boolean} notify force the onChange call
     */
    update(notify) {
        const m = this.evaluate();
        const q = (id) => this.container.querySelector(id);
        const pct = (v) => `${Math.round(v * 100)}%`;
        for (const key of ['w1', 'w2', 'b']) {
            q('#aisim-log-' + key).value = this[key];
            q('#aisim-log-' + key + '-val').textContent = this[key].toFixed(2);
        }
        q('#aisim-log-th-val').textContent = this.threshold.toFixed(2);
        q('#aisim-log-epoch').textContent = this.epoch;
        q('#aisim-log-loss').textContent = m.loss.toFixed(4);
        q('#aisim-log-acc').textContent = pct(m.accuracy);
        q('#aisim-log-prec').textContent = pct(m.precision);
        q('#aisim-log-rec').textContent = pct(m.recall);
        q('#aisim-log-advice').textContent = this.advice(m);
        this.showCalc();
        this.draw();

        const now = Date.now();
        if (typeof this.onChange === 'function' && (notify || now - this.lastNotify > 250)) {
            this.lastNotify = now;
            this.onChange({
                simtype: 'logistic',
                parameters: {
                    dataset: this.datasetKey,
                    w1: round(this.w1, 3),
                    w2: round(this.w2, 3),
                    b: round(this.b, 3),
                    threshold: this.threshold,
                    lr: this.lr,
                    epochs: this.epoch
                },
                metrics: {
                    loss: round(m.loss, 4),
                    accuracy: round(m.accuracy, 3),
                    precision: round(m.precision, 3),
                    recall: round(m.recall, 3)
                }
            });
        }
    }

    advice(m) {
        if (this.datasetKey === 'ring' && (this.epoch >= 50 || m.accuracy < 0.8) && this.w1 + this.w2 + this.b !== 0) {
            return this.t.adviceRing;
        }
        if (m.accuracy >= 0.88 && Math.abs(this.threshold - 0.5) >= 0.15) {
            return this.t.adviceThreshold;
        }
        if (m.accuracy >= 0.88) {
            return this.t.adviceGood;
        }
        return this.t.adviceStart;
    }

    draw() {
        this.drawMap();
        this.drawSigmoid();
        this.drawCurve();
    }

    drawMap() {
        const {w, h, ctx} = this.sizesPx.main;
        const toPx = (x1, x2) => ({px: ((x1 + RANGE) / (2 * RANGE)) * w, py: h - ((x2 + RANGE) / (2 * RANGE)) * h});
        ctx.clearRect(0, 0, w, h);
        const step = 8;
        for (let px = 0; px < w; px += step) {
            const x1 = -RANGE + ((px + step / 2) / w) * 2 * RANGE;
            for (let py = 0; py < h; py += step) {
                const x2 = RANGE - ((py + step / 2) / h) * 2 * RANGE;
                const p = sigmoid(this.w1 * x1 + this.w2 * x2 + this.b);
                const strength = Math.min(1, Math.abs(p - this.threshold) * 2) * 0.32 + 0.04;
                ctx.fillStyle = p >= this.threshold ? `rgba(239, 68, 68, ${strength})` : `rgba(37, 99, 235, ${strength})`;
                ctx.fillRect(px, py, step, step);
            }
        }

        // Axes through the origin.
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.18)';
        ctx.lineWidth = 1;
        const o = toPx(0, 0);
        ctx.beginPath();
        ctx.moveTo(0, o.py);
        ctx.lineTo(w, o.py);
        ctx.moveTo(o.px, 0);
        ctx.lineTo(o.px, h);
        ctx.stroke();

        this.drawBoundaryLine(ctx, toPx);

        this.points.forEach((pt, i) => {
            const {px, py} = toPx(pt.x1, pt.x2);
            const pred = sigmoid(this.z(pt)) >= this.threshold ? 1 : 0;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fillStyle = pt.cls === 1 ? '#ef4444' : '#2563eb';
            ctx.fill();
            ctx.lineWidth = pred === pt.cls ? 1.5 : 2.5;
            ctx.strokeStyle = pred === pt.cls ? '#ffffff' : '#0f172a';
            ctx.stroke();
            if (i === this.selected) {
                ctx.beginPath();
                ctx.arc(px, py, 9, 0, Math.PI * 2);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#f59e0b';
                ctx.stroke();
            }
        });

        const l = this.labels();
        ctx.fillStyle = '#475569';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`x₁: ${l.x1} →`, w - 6, h - 6);
        ctx.textAlign = 'left';
        ctx.fillText(`↑ x₂: ${l.x2}`, 6, 14);
    }

    /**
     * The decision boundary: the points where σ(z) = θ, i.e. w1·x1 + w2·x2 + b = ln(θ / (1 − θ)).
     *
     * @param {CanvasRenderingContext2D} ctx
     * @param {Function} toPx
     */
    drawBoundaryLine(ctx, toPx) {
        const c = Math.log(this.threshold / (1 - this.threshold)) - this.b;
        let a;
        let bEnd;
        if (Math.abs(this.w2) > 1e-6 && Math.abs(this.w2) >= Math.abs(this.w1)) {
            a = toPx(-RANGE, (c + this.w1 * RANGE) / this.w2);
            bEnd = toPx(RANGE, (c - this.w1 * RANGE) / this.w2);
        } else if (Math.abs(this.w1) > 1e-6) {
            a = toPx((c + this.w2 * RANGE) / this.w1, -RANGE);
            bEnd = toPx((c - this.w2 * RANGE) / this.w1, RANGE);
        } else {
            return;
        }
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(bEnd.px, bEnd.py);
        ctx.stroke();
    }

    drawSigmoid() {
        const {w, h, ctx} = this.sizesPx.sig;
        ctx.clearRect(0, 0, w, h);
        const pad = 14;
        const zmax = 8;
        const toPx = (z, p) => ({px: pad + ((z + zmax) / (2 * zmax)) * (w - 2 * pad), py: h - pad - p * (h - 2 * pad)});

        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const p of [0, 1]) {
            const a = toPx(-zmax, p);
            const b = toPx(zmax, p);
            ctx.moveTo(a.px, a.py);
            ctx.lineTo(b.px, b.py);
        }
        const z0 = toPx(0, 0);
        ctx.moveTo(z0.px, pad);
        ctx.lineTo(z0.px, h - pad);
        ctx.stroke();

        // Threshold line.
        ctx.strokeStyle = '#0f172a';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        const t0 = toPx(-zmax, this.threshold);
        const t1 = toPx(zmax, this.threshold);
        ctx.moveTo(t0.px, t0.py);
        ctx.lineTo(t1.px, t1.py);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
            const z = -zmax + (2 * zmax * i) / 120;
            const {px, py} = toPx(z, sigmoid(z));
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        this.points.forEach((pt, i) => {
            const z = clamp(this.z(pt), -zmax, zmax);
            const {px, py} = toPx(z, sigmoid(z));
            ctx.beginPath();
            ctx.arc(px, py, i === this.selected ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = pt.cls === 1 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(37, 99, 235, 0.8)';
            ctx.fill();
            if (i === this.selected) {
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#f59e0b';
                ctx.stroke();
            }
        });

        ctx.fillStyle = '#475569';
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`θ = ${this.threshold.toFixed(2)}`, pad + 2, t0.py - 4);
        ctx.textAlign = 'right';
        ctx.fillText('z', w - 2, h - pad - 2);
    }

    drawCurve() {
        const {w, h, ctx} = this.sizesPx.curve;
        ctx.clearRect(0, 0, w, h);
        if (this.history.length < 2) {
            return;
        }
        const max = Math.max(...this.history, 0.1);
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.history.forEach((v, i) => {
            const px = (i / (this.history.length - 1)) * (w - 8) + 4;
            const py = h - 6 - (v / max) * (h - 12);
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        });
        ctx.stroke();
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
LogisticSim.STRINGS = [
    'log_title', 'log_badge', 'log_model', 'log_w1', 'log_w2', 'log_bias', 'log_threshold', 'log_threshold_hint',
    'log_training', 'log_lr', 'log_train', 'log_pause', 'log_step', 'log_reset', 'log_metrics', 'log_epoch', 'log_loss',
    'log_accuracy', 'log_precision', 'log_recall', 'log_loss_curve', 'log_map_title', 'log_sigmoid_title', 'log_boundary',
    'log_calc', 'log_calc_hint', 'log_advice_start', 'log_advice_good', 'log_advice_threshold', 'log_advice_ring',
    'log_datasets_machine', 'log_datasets_exam', 'log_datasets_ring', 'log_calc_point', 'log_calc_pred', 'log_calc_true',
    'log_machine_x1', 'log_machine_x2', 'log_machine_cls1', 'log_machine_cls0',
    'log_exam_x1', 'log_exam_x2', 'log_exam_cls1', 'log_exam_cls0',
    'log_ring_x1', 'log_ring_x2', 'log_ring_cls1', 'log_ring_cls0'
];
