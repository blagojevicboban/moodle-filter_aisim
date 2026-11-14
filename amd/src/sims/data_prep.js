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
 * Data Preparation (EDA) interactive simulation engine.
 *
 * Raw training data with missing values, outliers and two variables on very different scales. The student explores it
 * (statistics and histograms) and builds a preparation pipeline – missing values, outliers, scaling – for a k-NN model
 * that is measured on clean test data. A view with the same scale on both axes shows the space the model really sees.
 *
 * @module filter_aisim/sims/data_prep
 * @copyright 2026 Boban Blagojević
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

import {renderTemplate, whenRendered, selectOptions} from 'filter_aisim/sims/layout';

const DATASETS = ['machine', 'patients'];
const MISSING = ['zero', 'drop', 'mean', 'median'];
const OUTLIERS = ['keep', 'remove'];
const SCALING = ['none', 'minmax', 'zscore'];

/**
 * How each dataset is generated: variable 1 hardly separates the classes but has a large range, variable 2 separates
 * them but has a small range. Missing values are only in variable 2, outliers only in variable 1 (a unit mistake).
 */
const SPEC = {
    machine: {
        n: [70, 50], test: [50, 40],
        x1: {mean: [1850, 2000], sd: 450, lo: 800, hi: 3000, digits: 0},
        x2: {mean: [1.6, 3.4], sd: 0.6, lo: 0.2, hi: 6, digits: 2},
        // A failing machine often loses its vibration sensor.
        missing: [0.04, 0.3],
        outliers: [24000, 31000, 38000, 45000],
    },
    patients: {
        n: [65, 55], test: [45, 45],
        x1: {mean: [48, 56], sd: 15, lo: 18, hi: 90, digits: 0},
        x2: {mean: [0.7, 1.35], sd: 0.25, lo: 0.1, hi: 2.4, digits: 2},
        missing: [0.15, 0.15],
        outliers: [520, 610, 700],
    },
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const round = (v, digits) => Math.round(v * Math.pow(10, digits)) / Math.pow(10, digits);
const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const median = (a) => {
    const s = [...a].sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const sd = (a) => {
    const m = mean(a);
    return Math.sqrt(mean(a.map((v) => (v - m) ** 2))) || 1;
};

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

export class DataPrepSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('dp_title'),
            badge: this.str('dp_badge'),
            edaTitle: this.str('dp_eda_title'),
            stat: this.str('dp_stat'),
            statMin: this.str('dp_stat_min'),
            statMax: this.str('dp_stat_max'),
            statMean: this.str('dp_stat_mean'),
            statMedian: this.str('dp_stat_median'),
            statMissing: this.str('dp_stat_missing'),
            histTitle: this.str('dp_hist_title'),
            spaceTitle: this.str('dp_space_title'),
            spaceHint: this.str('dp_space_hint'),
            pipeline: this.str('dp_pipeline'),
            missingLabel: this.str('dp_missing_label'),
            outliersLabel: this.str('dp_outliers_label'),
            scalingLabel: this.str('dp_scaling_label'),
            k: this.str('dp_k'),
            metrics: this.str('dp_metrics'),
            rows: this.str('dp_rows'),
            accuracy: this.str('dp_accuracy'),
            recall: this.str('dp_recall'),
            testPts: this.str('dp_test_pts'),
            wrongPts: this.str('dp_wrong_pts'),
            adviceStart: this.str('dp_advice_start'),
        };

        this.datasetKey = DATASETS.includes(options.dataset) ? options.dataset : 'machine';
        this.missing = MISSING.includes(options.missing) ? options.missing : 'zero';
        this.outliers = OUTLIERS.includes(options.outliers) ? options.outliers : 'keep';
        this.scaling = SCALING.includes(options.scaling) ? options.scaling : 'none';
        this.k = clamp(parseInt(options.k, 10) || 5, 1, 15);
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
     * Names of the two variables and the two classes of the dataset.
     *
     * @returns {Object}
     */
    labels() {
        if (this.datasetKey === 'patients') {
            return {x1: this.str('dp_patients_x1'), x2: this.str('dp_patients_x2'),
                cls0: this.str('dp_patients_cls0'), cls1: this.str('dp_patients_cls1')};
        }
        return {x1: this.str('dp_machine_x1'), x2: this.str('dp_machine_x2'),
            cls0: this.str('dp_machine_cls0'), cls1: this.str('dp_machine_cls1')};
    }

    /**
     * Raw training data (with missing values in x2 as null and outliers in x1) and clean test data.
     */
    initData() {
        const spec = SPEC[this.datasetKey];
        const rand = seeded(4000 + DATASETS.indexOf(this.datasetKey));
        const gauss = () => {
            const u = Math.max(rand(), 1e-9);
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
        };
        const value = (v, cls) => round(clamp(v.mean[cls] + gauss() * v.sd, v.lo, v.hi), v.digits);
        const make = (counts, dirty) => {
            const rows = [];
            counts.forEach((n, cls) => {
                for (let i = 0; i < n; i++) {
                    const x2 = value(spec.x2, cls);
                    rows.push({x1: value(spec.x1, cls), x2: dirty && rand() < spec.missing[cls] ? null : x2, cls});
                }
            });
            return rows;
        };
        this.raw = make(spec.n, true);
        // Unit mistakes: a few readings of variable 1 a thousand-fold or ten-fold too large.
        spec.outliers.forEach((v, i) => {
            this.raw[(i * 37) % this.raw.length].x1 = v;
        });
        this.test = make(spec.test, false);
    }

    /**
     * The pipeline: missing values, outliers, then scaling fitted on the training data and applied to the test data.
     */
    prepare() {
        let rows = this.raw.map((r) => ({...r}));
        const nMissing = rows.filter((r) => r.x2 === null).length;
        // Outliers are found with the median and the spread around it (robust to the outliers themselves).
        const med = median(rows.map((r) => r.x1));
        const mad = median(rows.map((r) => Math.abs(r.x1 - med))) || 1;
        const isOutlier = (r) => Math.abs(r.x1 - med) / mad > 6;
        const nOutliers = rows.filter(isOutlier).length;
        if (this.outliers === 'remove') {
            rows = rows.filter((r) => !isOutlier(r));
        }
        const known = rows.filter((r) => r.x2 !== null).map((r) => r.x2);
        if (this.missing === 'drop') {
            rows = rows.filter((r) => r.x2 !== null);
        } else {
            const fill = {zero: 0, mean: mean(known), median: median(known)}[this.missing];
            rows.forEach((r) => {
                if (r.x2 === null) {
                    r.x2 = fill;
                    r.filled = true;
                }
            });
        }
        this.clean = rows;
        this.nMissing = nMissing;
        this.nOutliers = nOutliers;

        const fit = (key) => {
            const v = rows.map((r) => r[key]);
            if (this.scaling === 'minmax') {
                const lo = Math.min(...v);
                const span = (Math.max(...v) - lo) || 1;
                return (x) => (x - lo) / span;
            }
            if (this.scaling === 'zscore') {
                const m = mean(v);
                const s = sd(v);
                return (x) => (x - m) / s;
            }
            return (x) => x;
        };
        const f1 = fit('x1');
        const f2 = fit('x2');
        this.trainSpace = rows.map((r) => ({u: f1(r.x1), v: f2(r.x2), cls: r.cls, filled: r.filled}));
        this.testSpace = this.test.map((r) => ({u: f1(r.x1), v: f2(r.x2), cls: r.cls}));
    }

    /**
     * Classify the test points with k-NN in the prepared space; accuracy and recall of class 1.
     *
     * @returns {Object}
     */
    evaluate() {
        let correct = 0;
        let tp = 0;
        let pos = 0;
        this.testSpace.forEach((p) => {
            const near = this.trainSpace
                .map((q) => ({d: (q.u - p.u) ** 2 + (q.v - p.v) ** 2, cls: q.cls}))
                .sort((a, b) => a.d - b.d)
                .slice(0, this.k);
            const votes = near.reduce((s, q) => s + q.cls, 0);
            p.pred = votes * 2 > near.length ? 1 : 0;
            correct += p.pred === p.cls ? 1 : 0;
            pos += p.cls;
            tp += p.pred === 1 && p.cls === 1 ? 1 : 0;
        });
        return {accuracy: correct / this.testSpace.length, recall: pos ? tp / pos : 0};
    }

    async renderLayout() {
        const l = this.labels();
        const context = {
            t: this.t,
            x1: l.x1,
            x2: l.x2,
            cls0: l.cls0,
            cls1: l.cls1,
            datasets: selectOptions([['machine', this.str('dp_datasets_machine')],
                ['patients', this.str('dp_datasets_patients')]], this.datasetKey),
            missing: selectOptions([['zero', this.str('dp_missing_zero')], ['drop', this.str('dp_missing_drop')],
                ['mean', this.str('dp_missing_mean')], ['median', this.str('dp_missing_median')]], this.missing),
            outliers: selectOptions([['keep', this.str('dp_outliers_keep')], ['remove', this.str('dp_outliers_remove')]],
                this.outliers),
            scaling: selectOptions([['none', this.str('dp_scaling_none')], ['minmax', this.str('dp_scaling_minmax')],
                ['zscore', this.str('dp_scaling_zscore')]], this.scaling),
            kslider: {id: 'aisim-dp-k', label: this.t.k, value: this.k, text: String(this.k), min: 1, max: 15, step: 2},
        };
        await renderTemplate(this.container, 'data_prep', context);
        this.canvas = this.container.querySelector('#aisim-dp-space');
        this.h1 = this.container.querySelector('#aisim-dp-h1');
        this.h2 = this.container.querySelector('#aisim-dp-h2');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv, fallbackH] of [['space', this.canvas, 300], ['h1', this.h1, 110], ['h2', this.h2, 110]]) {
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
        const controls = ['#aisim-dp-dataset', '#aisim-dp-missing', '#aisim-dp-outliers', '#aisim-dp-scaling', '#aisim-dp-k'];
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        q('#aisim-dp-dataset').addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.initData();
            whenRendered(this.renderLayout(), () => {
                this.bindEvents();
                this.update();
            });
        });
        for (const [id, key] of [['#aisim-dp-missing', 'missing'], ['#aisim-dp-outliers', 'outliers'],
                ['#aisim-dp-scaling', 'scaling']]) {
            q(id).addEventListener('change', (e) => {
                this[key] = e.target.value;
                this.update();
            });
        }
        q('#aisim-dp-k').addEventListener('input', (e) => {
            this.k = parseInt(e.target.value, 10);
            this.update();
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
    }

    update() {
        this.prepare();
        const m = this.evaluate();
        const q = (id) => this.container.querySelector(id);
        const spec = SPEC[this.datasetKey];
        q('#aisim-dp-k-val').textContent = this.k;
        q('#aisim-dp-rows').textContent = `${this.clean.length} / ${this.raw.length}`;
        q('#aisim-dp-acc').textContent = `${Math.round(m.accuracy * 100)}%`;
        q('#aisim-dp-rec').textContent = `${Math.round(m.recall * 100)}%`;
        // EDA of the training data after cleaning (before scaling).
        for (const [key, v] of [['x1', spec.x1], ['x2', spec.x2]]) {
            const col = this.clean.map((r) => r[key]);
            const f = (x) => x.toFixed(v.digits);
            q(`#aisim-dp-${key}-min`).textContent = f(Math.min(...col));
            q(`#aisim-dp-${key}-max`).textContent = f(Math.max(...col));
            q(`#aisim-dp-${key}-mean`).textContent = f(mean(col));
            q(`#aisim-dp-${key}-median`).textContent = f(median(col));
        }
        q('#aisim-dp-x1-missing').textContent = '0';
        // Missing values in the raw data; the pipeline decides what happens to them.
        q('#aisim-dp-x2-missing').textContent = String(this.nMissing);
        q('#aisim-dp-advice').textContent = this.advice(m);
        this.metrics = m;
        this.draw();

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'data_prep',
                parameters: {dataset: this.datasetKey, missing: this.missing, outliers: this.outliers, scaling: this.scaling,
                    k: this.k},
                metrics: {accuracy: round(m.accuracy, 3), recall: round(m.recall, 3), rows: this.clean.length}
            });
        }
    }

    advice(m) {
        if (this.scaling === 'none') {
            return this.str('dp_advice_scale');
        }
        if (this.scaling === 'minmax' && this.outliers === 'keep') {
            return this.str('dp_advice_outliers');
        }
        if (this.missing === 'zero') {
            return this.str('dp_advice_zero');
        }
        if (this.missing === 'drop' && m.recall < 0.85) {
            return this.str('dp_advice_drop');
        }
        if (m.accuracy >= 0.85) {
            return this.str('dp_advice_good');
        }
        return this.t.adviceStart;
    }

    draw() {
        this.drawHist('h1', 'x1');
        this.drawHist('h2', 'x2');
        this.drawSpace();
    }

    /**
     * Histogram of a variable of the cleaned training data, by class; filled values are drawn hatched.
     *
     * @param {String} key canvas key
     * @param {String} col 'x1' or 'x2'
     */
    drawHist(key, col) {
        const {w, h, ctx} = this.sizesPx[key];
        ctx.clearRect(0, 0, w, h);
        const vals = this.clean.map((r) => r[col]);
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        const bins = 24;
        const width = (hi - lo) / bins || 1;
        const counts = Array.from({length: bins}, () => [0, 0, 0]);
        this.clean.forEach((r) => {
            const b = clamp(Math.floor((r[col] - lo) / width), 0, bins - 1);
            counts[b][col === 'x2' && r.filled ? 2 : r.cls]++;
        });
        const top = Math.max(...counts.map((c) => c[0] + c[1] + c[2]), 1);
        const pad = 14;
        const bw = (w - 2 * pad) / bins;
        const colours = ['#2563eb', '#ef4444', '#f59e0b'];
        counts.forEach((c, b) => {
            let y = h - pad;
            c.forEach((n, i) => {
                const bh = (n / top) * (h - 2 * pad);
                ctx.fillStyle = colours[i];
                ctx.fillRect(pad + b * bw + 1, y - bh, bw - 2, bh);
                y -= bh;
            });
        });
        ctx.fillStyle = '#475569';
        ctx.font = '10px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(lo.toFixed(col === 'x1' ? 0 : 1), pad, h - 2);
        ctx.textAlign = 'right';
        ctx.fillText(hi.toFixed(col === 'x1' ? 0 : 1), w - pad, h - 2);
    }

    /**
     * The space the k-NN model measures distances in, with the same scale on both axes.
     */
    drawSpace() {
        const {w, h, ctx} = this.sizesPx.space;
        ctx.clearRect(0, 0, w, h);
        const all = this.trainSpace.concat(this.testSpace);
        const us = all.map((p) => p.u);
        const vs = all.map((p) => p.v);
        const umin = Math.min(...us);
        const vmin = Math.min(...vs);
        const span = Math.max(Math.max(...us) - umin, Math.max(...vs) - vmin) || 1;
        const pad = 16;
        const scale = Math.min(w - 2 * pad, h - 2 * pad) / span;
        const toPx = (u, v) => ({px: pad + (u - umin) * scale, py: h - pad - (v - vmin) * scale});

        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.strokeRect(pad, h - pad - span * scale, span * scale, span * scale);

        this.trainSpace.forEach((p) => {
            const {px, py} = toPx(p.u, p.v);
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = p.cls ? 'rgba(239, 68, 68, 0.75)' : 'rgba(37, 99, 235, 0.75)';
            ctx.fill();
            if (p.filled) {
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        });
        this.testSpace.forEach((p) => {
            const {px, py} = toPx(p.u, p.v);
            const right = p.pred === p.cls;
            const colour = p.cls ? '#ef4444' : '#2563eb';
            ctx.lineWidth = right ? 1.2 : 2.4;
            ctx.strokeStyle = right ? colour : '#0f172a';
            ctx.strokeRect(px - 3.5, py - 3.5, 7, 7);
        });

        const l = this.labels();
        ctx.fillStyle = '#475569';
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${l.x1} →`, w - 4, h - 3);
        ctx.textAlign = 'left';
        ctx.fillText(`↑ ${l.x2}`, 4, 12);
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
DataPrepSim.STRINGS = [
    'dp_title', 'dp_badge', 'dp_eda_title', 'dp_stat', 'dp_stat_min', 'dp_stat_max', 'dp_stat_mean', 'dp_stat_median',
    'dp_stat_missing', 'dp_hist_title', 'dp_space_title', 'dp_space_hint', 'dp_pipeline', 'dp_missing_label',
    'dp_outliers_label', 'dp_scaling_label', 'dp_k', 'dp_metrics', 'dp_rows', 'dp_accuracy', 'dp_recall', 'dp_test_pts',
    'dp_wrong_pts', 'dp_advice_start', 'dp_advice_scale', 'dp_advice_outliers', 'dp_advice_zero', 'dp_advice_drop',
    'dp_advice_good', 'dp_datasets_machine', 'dp_datasets_patients', 'dp_missing_zero', 'dp_missing_drop',
    'dp_missing_mean', 'dp_missing_median', 'dp_outliers_keep', 'dp_outliers_remove', 'dp_scaling_none',
    'dp_scaling_minmax', 'dp_scaling_zscore', 'dp_machine_x1', 'dp_machine_x2', 'dp_machine_cls0', 'dp_machine_cls1',
    'dp_patients_x1', 'dp_patients_x2', 'dp_patients_cls0', 'dp_patients_cls1'
];
