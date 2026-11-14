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
 * Recurrent Neural Network interactive simulation engine.
 *
 * One recurrent neuron reads a sequence step by step: h_t = tanh(w·h_(t-1) + u·x_t), y_t = v·h_t. The weight w decides
 * how much of the previous state (the memory) is kept. The student sets w, u and v or trains them with gradient descent,
 * sees the sequence, the network unrolled in time and how strongly each past input still influences the last output,
 * which shows forgetting and the vanishing gradient (the reason for LSTM and GRU networks).
 *
 * @module filter_aisim/sims/rnn
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {renderTemplate, whenRendered, selectOptions} from 'filter_aisim/sims/layout';

const DATASETS = ['temperature', 'memory'];
/** Hours of the temperature series. */
const HOURS = 72;
/** Pulses of the memory task: one sequence per pulse. */
const PULSES = [0.85, -0.55, 0.35, -0.8, 0.6, -0.1, 0.15, -0.3];
/** Allowed range of each weight (also the slider range). */
const RANGE = {w: [-1.5, 1.5], u: [-3, 3], v: [-4, 4]};
const LEARNING_RATE = 0.5;
/** Steps of the network drawn unrolled in time. */
const UNROLLED = 5;
/** An input counts as remembered while its influence is at least this share of the strongest one. */
const REMEMBERED = 0.1;

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
 * Run the recurrent neuron over a sequence.
 *
 * @param {Object} p Weights {w, u, v}
 * @param {Array} xs Inputs
 * @returns {Array} States h_t
 */
const unroll = (p, xs) => {
    let h = 0;
    return xs.map((x) => {
        h = Math.tanh(p.w * h + p.u * x);
        return h;
    });
};

export class RnnSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('rnn_title'),
            badge: this.str('rnn_badge'),
            seqTitle: this.str('rnn_seq_title'),
            input: this.str('rnn_input'),
            state: this.str('rnn_state'),
            output: this.str('rnn_output'),
            target: this.str('rnn_target'),
            example: this.str('rnn_example'),
            unrolledTitle: this.str('rnn_unrolled_title'),
            unrolledHint: this.str('rnn_unrolled_hint'),
            influenceTitle: this.str('rnn_influence_title'),
            influenceHint: this.str('rnn_influence_hint'),
            weights: this.str('rnn_weights'),
            formula: this.str('rnn_formula'),
            training: this.str('rnn_training'),
            train: this.str('rnn_train'),
            reset: this.str('rnn_reset'),
            metrics: this.str('rnn_metrics'),
            epoch: this.str('rnn_epoch'),
            loss: this.str('rnn_loss'),
            remembered: this.str('rnn_remembered'),
            gradient: this.str('rnn_gradient'),
            adviceStart: this.str('rnn_advice_start'),
        };

        this.datasetKey = DATASETS.includes(options.dataset) ? options.dataset : 'temperature';
        this.length = clamp(parseInt(options.length, 10) || 10, 3, 30);
        const num = (value, fallback, key) => {
            const n = parseFloat(value);
            return clamp(isFinite(n) ? n : fallback, RANGE[key][0], RANGE[key][1]);
        };
        this.p = {w: num(options.w, 0.5, 'w'), u: num(options.u, 1, 'u'), v: num(options.v, 1, 'v')};
        this.start = {...this.p};
        this.epoch = 0;
        this.exampleIndex = 0;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.initData();
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
     * Temperature: a noisy sensor over three days, the target is the real temperature of the next hour.
     * Memory: a pulse at the first step, then zeros; the target at the last step is the pulse.
     */
    initData() {
        const rand = seeded(5001);
        const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand());
        const real = Array.from({length: HOURS + 1}, (_, t) => 0.7 * Math.sin(2 * Math.PI * t / 24));
        const sensor = real.map((r) => r + 0.3 * gauss());
        this.temperature = {xs: sensor.slice(0, HOURS), targets: real.slice(1)};
    }

    /**
     * The sequences the loss is measured on: [{xs, targets}], a target of null is not scored.
     *
     * @returns {Array}
     */
    sequences() {
        if (this.datasetKey === 'temperature') {
            // The first three hours are skipped: the state needs a few steps to warm up.
            return [{xs: this.temperature.xs, targets: this.temperature.targets.map((v, t) => (t < 3 ? null : v))}];
        }
        return PULSES.map((a) => ({
            xs: [a, ...Array(this.length - 1).fill(0)],
            targets: [...Array(this.length - 1).fill(null), a],
        }));
    }

    /**
     * Mean squared error of the outputs against the targets.
     *
     * @param {Object} p Weights
     * @returns {Number}
     */
    loss(p) {
        let sum = 0;
        let n = 0;
        this.sequences().forEach((s) => {
            unroll(p, s.xs).forEach((h, t) => {
                if (s.targets[t] !== null) {
                    sum += (p.v * h - s.targets[t]) ** 2;
                    n++;
                }
            });
        });
        return sum / (n || 1);
    }

    /**
     * Numerical gradient of the loss with respect to w, u and v.
     *
     * @returns {Object}
     */
    gradient() {
        const e = 1e-4;
        const g = {};
        for (const k of ['w', 'u', 'v']) {
            const up = {...this.p, [k]: this.p[k] + e};
            const down = {...this.p, [k]: this.p[k] - e};
            g[k] = (this.loss(up) - this.loss(down)) / (2 * e);
        }
        return g;
    }

    /**
     * Gradient descent on w, u and v.
     *
     * @param {Number} count Epochs
     */
    train(count) {
        for (let i = 0; i < count; i++) {
            const g = this.gradient();
            for (const k of ['w', 'u', 'v']) {
                this.p[k] = clamp(this.p[k] - LEARNING_RATE * g[k], RANGE[k][0], RANGE[k][1]);
            }
            this.epoch++;
        }
    }

    /**
     * The sequence shown in the charts: the temperature series, or one example of the memory task.
     *
     * @returns {Object}
     */
    shown() {
        const all = this.sequences();
        return all[this.exampleIndex % all.length];
    }

    /**
     * Relative influence of each past input on the last output, |∂y_T/∂x_t| divided by the largest one, from the
     * newest step backwards (index 0 = the last step).
     *
     * @param {Array} xs Inputs
     * @param {Array} hs States
     * @returns {Array}
     */
    influence(xs, hs) {
        const n = xs.length;
        const window = this.datasetKey === 'temperature' ? 24 : n;
        const inf = [];
        let g = this.p.v * (1 - hs[n - 1] ** 2);
        for (let t = n - 1; t >= n - window; t--) {
            inf.push(Math.abs(g * this.p.u));
            if (t > 0) {
                g *= this.p.w * (1 - hs[t - 1] ** 2);
            }
        }
        const top = Math.max(...inf);
        return inf.map((v) => (top > 0 ? v / top : 0));
    }

    async renderLayout() {
        const slider = (key, label, step) => ({id: 'aisim-rnn-' + key, label, value: this.p[key],
            text: this.p[key].toFixed(2), min: RANGE[key][0], max: RANGE[key][1], step});
        const context = {
            t: this.t,
            memory: this.datasetKey === 'memory',
            datasets: selectOptions([['temperature', this.str('rnn_datasets_temperature')],
                ['memory', this.str('rnn_datasets_memory')]], this.datasetKey),
            lengthslider: {id: 'aisim-rnn-length', label: this.str('rnn_length'), value: this.length,
                text: String(this.length), min: 3, max: 30, step: 1},
            wslider: slider('w', this.str('rnn_w'), 0.05),
            uslider: slider('u', this.str('rnn_u'), 0.1),
            vslider: slider('v', this.str('rnn_v'), 0.1),
        };
        await renderTemplate(this.container, 'rnn', context);
        this.canvases = {
            seq: this.container.querySelector('#aisim-rnn-seq'),
            cell: this.container.querySelector('#aisim-rnn-cell'),
            inf: this.container.querySelector('#aisim-rnn-inf'),
        };
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, fallbackH] of [['seq', 220], ['cell', 170], ['inf', 120]]) {
            const cv = this.canvases[key];
            const rect = cv.getBoundingClientRect();
            const w = rect.width || 300;
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
        // Another example only changes the view, so it also works in a quiz review.
        q('#aisim-rnn-next').addEventListener('click', () => {
            this.exampleIndex = (this.exampleIndex + 1) % PULSES.length;
            this.update();
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
        const controls = ['#aisim-rnn-dataset', '#aisim-rnn-length', '#aisim-rnn-w', '#aisim-rnn-u', '#aisim-rnn-v',
            '#aisim-rnn-train', '#aisim-rnn-reset'];
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        q('#aisim-rnn-dataset').addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.exampleIndex = 0;
            this.epoch = 0;
            whenRendered(this.renderLayout(), () => {
                this.bindEvents();
                this.update();
            });
        });
        q('#aisim-rnn-length').addEventListener('input', (e) => {
            this.length = parseInt(e.target.value, 10);
            this.update();
        });
        for (const k of ['w', 'u', 'v']) {
            q('#aisim-rnn-' + k).addEventListener('input', (e) => {
                this.p[k] = parseFloat(e.target.value);
                this.update();
            });
        }
        q('#aisim-rnn-train').addEventListener('click', () => {
            this.train(10);
            this.update();
        });
        q('#aisim-rnn-reset').addEventListener('click', () => {
            this.p = {...this.start};
            this.epoch = 0;
            this.update();
        });
    }

    /**
     * Refresh the sliders, numbers and drawings and tell the listener (quiz answer).
     */
    update() {
        const q = (id) => this.container.querySelector(id);
        for (const k of ['w', 'u', 'v']) {
            q('#aisim-rnn-' + k).value = this.p[k];
            q('#aisim-rnn-' + k + '-val').textContent = this.p[k].toFixed(2);
        }
        q('#aisim-rnn-length-val').textContent = this.length;
        const loss = this.loss(this.p);
        const g = this.gradient();
        const gsize = Math.sqrt(g.w ** 2 + g.u ** 2 + g.v ** 2);
        const s = this.shown();
        const hs = unroll(this.p, s.xs);
        const inf = this.influence(s.xs, hs);
        // Count the steps back to the oldest input that still has a noticeable influence.
        let remembered = 0;
        inf.forEach((v, i) => {
            if (v >= REMEMBERED) {
                remembered = i + 1;
            }
        });
        this.view = {s, hs, inf};
        this.state = {loss, gsize, remembered};

        q('#aisim-rnn-epoch').textContent = this.epoch;
        q('#aisim-rnn-loss').textContent = loss.toFixed(4);
        q('#aisim-rnn-mem').textContent = String(remembered);
        q('#aisim-rnn-grad').textContent = gsize < 1e-4 ? '≈ 0' : gsize.toFixed(4);
        q('#aisim-rnn-advice').textContent = this.advice();
        this.draw();

        if (typeof this.onChange === 'function') {
            const parameters = {dataset: this.datasetKey, w: round(this.p.w, 3), u: round(this.p.u, 3),
                v: round(this.p.v, 3)};
            if (this.datasetKey === 'memory') {
                parameters.length = this.length;
            }
            this.onChange({
                simtype: 'rnn',
                parameters,
                metrics: {loss: round(loss, 4), remembered, epochs: this.epoch}
            });
        }
    }

    /**
     * A hint for the current weights.
     *
     * @returns {String}
     */
    advice() {
        const {loss, gsize} = this.state;
        const w = this.p.w;
        if (this.datasetKey === 'memory') {
            if (loss < 0.02) {
                return this.str('rnn_advice_mem_good');
            }
            if (gsize < 1e-3 && this.epoch > 0) {
                return this.str('rnn_advice_vanish');
            }
            if (w > 1.05) {
                return this.str('rnn_advice_saturate');
            }
            if (w < 0.8) {
                return this.str('rnn_advice_fade');
            }
            return this.str('rnn_advice_squash');
        }
        if (loss < 0.065) {
            return this.str('rnn_advice_temp_good');
        }
        if (w < -0.1) {
            return this.str('rnn_advice_negative');
        }
        if (Math.abs(w) <= 0.1) {
            return this.str('rnn_advice_nomemory');
        }
        if (w > 0.8) {
            return this.str('rnn_advice_slow');
        }
        return this.t.adviceStart;
    }

    draw() {
        this.drawSequence();
        this.drawCell();
        this.drawInfluence();
    }

    /**
     * Inputs (dots), state h, output y and target over the steps of the shown sequence.
     */
    drawSequence() {
        const {w, h, ctx} = this.sizesPx.seq;
        ctx.clearRect(0, 0, w, h);
        const {s, hs} = this.view;
        const n = s.xs.length;
        const pad = 18;
        const toX = (t) => pad + (n > 1 ? t / (n - 1) : 0.5) * (w - 2 * pad);
        const toY = (v) => h / 2 - clamp(v, -1.4, 1.4) / 1.4 * (h / 2 - pad);

        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad, toY(0));
        ctx.lineTo(w - pad, toY(0));
        ctx.stroke();

        s.xs.forEach((x, t) => {
            ctx.beginPath();
            ctx.arc(toX(t), toY(x), 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#94a3b8';
            ctx.fill();
        });
        const line = (values, colour, width, dash) => {
            ctx.strokeStyle = colour;
            ctx.lineWidth = width;
            ctx.setLineDash(dash);
            ctx.beginPath();
            values.forEach((v, t) => {
                if (t === 0) {
                    ctx.moveTo(toX(t), toY(v));
                } else {
                    ctx.lineTo(toX(t), toY(v));
                }
            });
            ctx.stroke();
            ctx.setLineDash([]);
        };
        line(hs, '#a855f7', 1.5, [3, 3]);
        line(hs.map((v) => this.p.v * v), '#2563eb', 2.2, []);
        if (this.datasetKey === 'temperature') {
            line(s.targets.map((v, t) => (v === null ? this.temperature.targets[t] : v)), '#16a34a', 1.5, [6, 4]);
        } else {
            const t = n - 1;
            ctx.strokeStyle = '#16a34a';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(toX(t) - 7, toY(s.targets[t]));
            ctx.lineTo(toX(t) + 7, toY(s.targets[t]));
            ctx.stroke();
        }
        ctx.fillStyle = '#475569';
        ctx.font = '10px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('t = 1', pad, h - 3);
        ctx.textAlign = 'right';
        ctx.fillText('t = ' + n, w - pad, h - 3);
    }

    /**
     * The network unrolled over the last steps: inputs below, states in the middle, outputs above. The width of an
     * arrow follows the size of its weight; red means a negative weight.
     */
    drawCell() {
        const {w, h, ctx} = this.sizesPx.cell;
        ctx.clearRect(0, 0, w, h);
        const {s, hs} = this.view;
        const n = s.xs.length;
        const first = Math.max(0, n - UNROLLED);
        const steps = n - first;
        const gap = (w - 20) / steps;
        const cx = (i) => 10 + gap * (i + 0.5);
        const r = Math.min(20, gap / 2 - 8);
        const yIn = h - 22;
        const yH = h / 2 + 4;
        const yOut = 24;
        const weightStyle = (v) => {
            ctx.strokeStyle = v < 0 ? '#ef4444' : '#2563eb';
            ctx.fillStyle = ctx.strokeStyle;
            ctx.lineWidth = 0.5 + Math.min(4, Math.abs(v) * 2.5);
        };
        const arrow = (x1, y1, x2, y2, v) => {
            weightStyle(v);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            const a = Math.atan2(y2 - y1, x2 - x1);
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4));
            ctx.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4));
            ctx.closePath();
            ctx.fill();
        };
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        for (let i = 0; i < steps; i++) {
            const t = first + i;
            const x = cx(i);
            arrow(x, yIn - 8, x, yH + r + 2, this.p.u);
            arrow(x, yH - r - 2, x, yOut + 10, this.p.v);
            if (i > 0 || first > 0) {
                const from = i > 0 ? cx(i - 1) + r + 2 : x - gap / 2;
                arrow(from, yH, x - r - 2, yH, this.p.w);
            }
            // The state circle: its fill shows the value of h (blue positive, red negative).
            const v = hs[t];
            ctx.beginPath();
            ctx.arc(x, yH, r, 0, Math.PI * 2);
            ctx.fillStyle = v < 0 ? `rgba(239, 68, 68, ${0.15 + 0.6 * Math.abs(v)})`
                : `rgba(168, 85, 247, ${0.15 + 0.6 * Math.abs(v)})`;
            ctx.fill();
            ctx.strokeStyle = '#475569';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = '#0f172a';
            ctx.fillText(v.toFixed(2), x, yH + 4);
            ctx.fillStyle = '#475569';
            const xText = s.xs[t].toFixed(2);
            const yText = (this.p.v * v).toFixed(2);
            if (gap < 90) {
                // Narrow screens: the name above its value, so that neighbouring labels do not overlap.
                ctx.fillText(`x${t + 1}`, x, h - 16);
                ctx.fillText(xText, x, h - 5);
                ctx.fillText(`y${t + 1}`, x, yOut - 11);
                ctx.fillText(yText, x, yOut);
            } else {
                ctx.fillText(`x${t + 1} = ${xText}`, x, h - 6);
                ctx.fillText(`y${t + 1} = ${yText}`, x, yOut);
            }
        }
        ctx.textAlign = 'left';
        ctx.fillStyle = '#475569';
        ctx.fillText(`w = ${this.p.w.toFixed(2)}`, 4, yH - r - 6);
    }

    /**
     * Bars of how strongly each past input still moves the last output (newest on the right).
     */
    drawInfluence() {
        const {w, h, ctx} = this.sizesPx.inf;
        ctx.clearRect(0, 0, w, h);
        const inf = this.view.inf;
        const n = inf.length;
        const pad = 16;
        const bw = (w - 2 * pad) / n;
        const base = h - pad;
        ctx.strokeStyle = '#cbd5e1';
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(pad, base - REMEMBERED * (base - 6));
        ctx.lineTo(w - pad, base - REMEMBERED * (base - 6));
        ctx.stroke();
        ctx.setLineDash([]);
        inf.forEach((v, back) => {
            const i = n - 1 - back;
            const bh = v * (base - 6);
            ctx.fillStyle = v >= REMEMBERED ? '#a855f7' : '#cbd5e1';
            ctx.fillRect(pad + i * bw + 1, base - bh, Math.max(1, bw - 2), bh);
        });
        ctx.fillStyle = '#475569';
        ctx.font = '10px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('t − ' + (n - 1), pad, h - 3);
        ctx.textAlign = 'right';
        ctx.fillText('t', w - pad, h - 3);
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
RnnSim.STRINGS = [
    'rnn_title', 'rnn_badge', 'rnn_seq_title', 'rnn_input', 'rnn_state', 'rnn_output', 'rnn_target', 'rnn_example',
    'rnn_unrolled_title', 'rnn_unrolled_hint', 'rnn_influence_title', 'rnn_influence_hint', 'rnn_weights',
    'rnn_formula', 'rnn_training', 'rnn_train', 'rnn_reset', 'rnn_metrics', 'rnn_epoch', 'rnn_loss', 'rnn_remembered',
    'rnn_gradient', 'rnn_advice_start', 'rnn_length', 'rnn_w', 'rnn_u', 'rnn_v',
    'rnn_datasets_temperature', 'rnn_datasets_memory', 'rnn_advice_mem_good', 'rnn_advice_vanish',
    'rnn_advice_saturate', 'rnn_advice_fade', 'rnn_advice_temp_good', 'rnn_advice_negative', 'rnn_advice_nomemory',
    'rnn_advice_slow', 'rnn_advice_squash'
];
