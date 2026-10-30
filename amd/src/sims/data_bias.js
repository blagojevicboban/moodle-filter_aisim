/**
 * Data split & bias: a k-NN model checks parts from two factories, A and B, where the defect rule differs.
 *
 * The student chooses how many samples to collect, how much of the training data comes from factory B, how much is
 * held out for testing and the k of the model. The model is then measured on the held-out test set and on a large
 * representative "real world" set, per factory: a sample with little B data looks fine on its own test set but fails
 * for factory B.
 *
 * @module filter_aisim/sims/data_bias
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

const POPULATIONS = ['shifted', 'similar'];
const WORLD_PER_GROUP = 300;
const GRID = 44;

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

const clampInt = (v, lo, hi, fallback) => {
    const n = parseInt(v, 10);
    return isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

export class DataBiasSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('db_title'),
            badge: this.str('db_badge'),
            story: this.str('db_story'),
            data: this.str('db_data'),
            n: this.str('db_n'),
            share: this.str('db_share'),
            test: this.str('db_test'),
            model: this.str('db_model'),
            k: this.str('db_k'),
            resample: this.str('db_resample'),
            metrics: this.str('db_metrics'),
            train: this.str('db_train'),
            testSet: this.str('db_test_set'),
            worldA: this.str('db_world_a'),
            worldB: this.str('db_world_b'),
            counts: this.str('db_counts'),
            gap: this.str('db_gap'),
            legend: this.str('db_legend'),
            populations: {
                shifted: this.str('db_populations_shifted'),
                similar: this.str('db_populations_similar')
            },
            adviceStart: this.str('db_advice_start'),
            adviceMemorize: this.str('db_advice_memorize'),
            adviceBias: this.str('db_advice_bias'),
            adviceSmallTest: this.str('db_advice_small_test'),
            adviceFair: this.str('db_advice_fair')
        };

        this.population = POPULATIONS.includes(options.population) ? options.population : 'shifted';
        this.n = clampInt(options.n, 40, 400, 200);
        this.share = clampInt(options.share, 0, 50, 5);
        this.testPct = clampInt(options.test, 10, 50, 25);
        this.k = clampInt(options.k, 1, 25, 5);
        this.seed = parseInt(options.seed, 10) || 1;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.initWorld();
        this.sample();
        this.renderLayout();
        this.bindEvents();
        this.update();
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
     * The true defect threshold for the second sensor at a given first sensor reading.
     *
     * @param {Number} x
     * @returns {Number}
     */
    rule(x) {
        if (this.population === 'similar') {
            return 0.35 + 0.15 * x;
        }
        // Factory B's parts (right side) are defective already at much higher readings.
        const t = Math.min(1, Math.max(0, (x - 0.58) / 0.1));
        return 0.25 + 0.53 * t * t * (3 - 2 * t);
    }

    /**
     * One part of a factory, with 3% label noise.
     *
     * @param {Function} rand
     * @param {String} group 'A' or 'B'
     * @returns {Object}
     */
    part(rand, group) {
        const x = group === 'A' ? rand() * 0.55 : 0.5 + rand() * 0.5;
        const y = rand();
        let label = y > this.rule(x) ? 1 : 0;
        if (rand() < 0.03) {
            label = 1 - label;
        }
        return {x, y, group, label};
    }

    initWorld() {
        const rand = seeded(424242 + POPULATIONS.indexOf(this.population));
        this.world = [];
        for (let i = 0; i < WORLD_PER_GROUP; i++) {
            this.world.push(this.part(rand, 'A'));
            this.world.push(this.part(rand, 'B'));
        }
    }

    sample() {
        const rand = seeded(this.seed * 7919 + POPULATIONS.indexOf(this.population));
        const nB = Math.round(this.n * this.share / 100);
        const pts = [];
        for (let i = 0; i < this.n; i++) {
            pts.push(this.part(rand, i < nB ? 'B' : 'A'));
        }
        // Shuffle, then hold out the last part for testing.
        for (let i = pts.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [pts[i], pts[j]] = [pts[j], pts[i]];
        }
        const nTest = Math.max(1, Math.round(this.n * this.testPct / 100));
        this.trainSet = pts.slice(0, pts.length - nTest);
        this.testSet = pts.slice(pts.length - nTest);
    }

    /**
     * Vote of the k nearest training points (k-NN).
     *
     * @param {Number} x
     * @param {Number} y
     * @returns {Number} share of "defective" among the k nearest neighbours
     */
    predict(x, y) {
        const k = Math.min(this.k, this.trainSet.length);
        const best = [];
        for (const p of this.trainSet) {
            const d = (p.x - x) ** 2 + (p.y - y) ** 2;
            if (best.length < k || d < best[best.length - 1].d) {
                let i = best.length - 1;
                if (best.length < k) {
                    best.push(null);
                    i = best.length - 1;
                }
                while (i > 0 && best[i - 1].d > d) {
                    best[i] = best[i - 1];
                    i--;
                }
                best[i] = {d, label: p.label};
            }
        }
        return best.reduce((a, b) => a + b.label, 0) / best.length;
    }

    accuracy(points) {
        if (!points.length) {
            return 0;
        }
        let ok = 0;
        for (const p of points) {
            if ((this.predict(p.x, p.y) > 0.5 ? 1 : 0) === p.label) {
                ok++;
            }
        }
        return ok / points.length;
    }

    evaluate() {
        return {
            trainAcc: this.accuracy(this.trainSet),
            testAcc: this.accuracy(this.testSet),
            accA: this.accuracy(this.world.filter((p) => p.group === 'A')),
            accB: this.accuracy(this.world.filter((p) => p.group === 'B'))
        };
    }

    renderLayout() {
        const opt = (value, label, current) => `<option value="${value}" ${value === current ? 'selected' : ''}>${label}</option>`;
        const slider = (id, label, value, min, max, step, unit = '') => `
            <div class="aisim-slider-group">
                <label for="aisim-db-${id}">${label} = <span id="aisim-db-${id}-val">${value}${unit}</span></label>
                <input type="range" class="aisim-slider" id="aisim-db-${id}" min="${min}" max="${max}" step="${step}"
                       value="${value}">
            </div>`;
        const stat = (id, label) => `<div class="aisim-stat-row"><span>${label}</span>
            <span class="aisim-stat-val" id="aisim-db-${id}">–</span></div>`;
        this.container.innerHTML = `
            <div class="aisim-container aisim-db">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <select class="aisim-select" id="aisim-db-population">
                        ${POPULATIONS.map((k) => opt(k, this.t.populations[k], this.population)).join('')}
                    </select>
                </div>

                <div class="aisim-body">
                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-ql-legend">${this.t.story}</div>
                            <div class="aisim-canvas-box" style="padding:0; overflow:hidden; min-height:0;">
                                <canvas class="aisim-canvas" id="aisim-db-canvas" style="height:320px;"></canvas>
                            </div>
                            <div class="aisim-ql-legend">${this.t.legend}</div>
                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-db-advice">${this.t.adviceStart}</div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.data}</div>
                                ${slider('n', this.t.n, this.n, 40, 400, 20)}
                                ${slider('share', this.t.share, this.share, 0, 50, 5, '%')}
                                ${slider('test', this.t.test, this.testPct, 10, 50, 5, '%')}
                                <div class="aisim-card-title" style="margin-top:0.75rem;">${this.t.model}</div>
                                ${slider('k', this.t.k, this.k, 1, 25, 2)}
                                <div class="aisim-controls">
                                    <button class="aisim-btn aisim-btn-secondary" id="aisim-db-resample">${this.t.resample}</button>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.metrics}</div>
                                <canvas class="aisim-canvas" id="aisim-db-bars" style="height:150px; cursor:default;"></canvas>
                                ${stat('counts', this.t.counts)}
                                ${stat('gap', this.t.gap)}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.canvas = this.container.querySelector('#aisim-db-canvas');
        this.bars = this.container.querySelector('#aisim-db-bars');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv, fallbackH] of [['main', this.canvas, 320], ['bars', this.bars, 150]]) {
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
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
        const ids = ['#aisim-db-population', '#aisim-db-n', '#aisim-db-share', '#aisim-db-test', '#aisim-db-k',
            '#aisim-db-resample'];
        if (this.readonly) {
            ids.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        q('#aisim-db-population').addEventListener('change', (e) => {
            this.population = e.target.value;
            this.initWorld();
            this.sample();
            this.update();
        });
        const sliders = {n: ['n', ''], share: ['share', '%'], test: ['testPct', '%'], k: ['k', '']};
        for (const [id, [prop, unit]] of Object.entries(sliders)) {
            q(`#aisim-db-${id}`).addEventListener('input', (e) => {
                this[prop] = parseInt(e.target.value, 10);
                q(`#aisim-db-${id}-val`).textContent = `${this[prop]}${unit}`;
                if (prop !== 'k') {
                    this.sample();
                }
                this.update();
            });
        }
        q('#aisim-db-resample').addEventListener('click', () => {
            this.seed++;
            this.sample();
            this.update();
        });
    }

    update() {
        const m = this.evaluate();
        this.metrics = m;
        const q = (id) => this.container.querySelector(id);
        const pct = (v) => `${Math.round(v * 100)}%`;
        q('#aisim-db-counts').textContent = `${this.trainSet.length} / ${this.testSet.length}`;
        q('#aisim-db-gap').textContent = pct(Math.abs(m.accA - m.accB));

        let advice = this.t.adviceStart;
        const worst = Math.min(m.accA, m.accB);
        if (worst >= 0.88 && Math.abs(m.accA - m.accB) <= 0.06) {
            advice = this.t.adviceFair;
        } else if (m.accA - m.accB > 0.1) {
            advice = this.t.adviceBias;
        } else if (this.k === 1) {
            advice = this.t.adviceMemorize;
        } else if (this.testSet.length < 25) {
            advice = this.t.adviceSmallTest;
        }
        q('#aisim-db-advice').textContent = advice;

        this.draw();

        if (typeof this.onChange === 'function') {
            const r = (v) => Math.round(v * 1000) / 1000;
            this.onChange({
                simtype: 'data_bias',
                parameters: {
                    population: this.population,
                    n: this.n,
                    share: this.share,
                    test: this.testPct,
                    k: this.k,
                    seed: this.seed
                },
                metrics: {
                    trainAcc: r(m.trainAcc),
                    testAcc: r(m.testAcc),
                    accA: r(m.accA),
                    accB: r(m.accB),
                    worst: r(worst),
                    gap: r(Math.abs(m.accA - m.accB))
                }
            });
        }
    }

    draw() {
        this.drawData();
        this.drawBars();
    }

    drawData() {
        const {w, h, ctx} = this.sizesPx.main;
        ctx.clearRect(0, 0, w, h);
        const toPx = (x, y) => ({px: x * w, py: h - y * h});
        const cw = w / GRID;
        const ch = h / GRID;
        // Whole-pixel cell edges: overlapping translucent cells would show seams.
        for (let i = 0; i < GRID; i++) {
            const x0 = Math.round(i * cw);
            const x1 = Math.round((i + 1) * cw);
            for (let j = 0; j < GRID; j++) {
                const y0 = Math.round(j * ch);
                const y1 = Math.round((j + 1) * ch);
                const p = this.predict((i + 0.5) / GRID, 1 - (j + 0.5) / GRID);
                ctx.fillStyle = p > 0.5 ? 'rgba(239, 68, 68, 0.16)' : 'rgba(37, 99, 235, 0.12)';
                ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
            }
        }
        // True rule.
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.6)';
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= 100; i++) {
            const {px, py} = toPx(i / 100, this.rule(i / 100));
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        const mark = (p, filled) => {
            const {px, py} = toPx(p.x, p.y);
            const colour = p.label ? '#ef4444' : '#2563eb';
            ctx.beginPath();
            if (p.group === 'A') {
                ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            } else {
                ctx.rect(px - 4.5, py - 4.5, 9, 9);
            }
            ctx.fillStyle = filled ? colour : '#ffffff';
            ctx.fill();
            ctx.strokeStyle = filled ? '#ffffff' : colour;
            ctx.lineWidth = filled ? 1.2 : 2;
            ctx.stroke();
        };
        this.trainSet.forEach((p) => mark(p, true));
        this.testSet.forEach((p) => mark(p, false));
        // Factory labels on a light backing, above the points.
        ctx.font = '600 12px sans-serif';
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        const label = (text, x) => {
            const tw = ctx.measureText(text).width;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.fillRect(x - 4, 4, tw + 8, 18);
            ctx.fillStyle = '#334155';
            ctx.fillText(text, x, 13);
        };
        label(this.str('db_factory_a'), 8);
        const textB = this.str('db_factory_b');
        label(textB, w - 8 - ctx.measureText(textB).width);
    }

    drawBars() {
        const {w, h, ctx} = this.sizesPx.bars;
        ctx.clearRect(0, 0, w, h);
        const m = this.metrics;
        const items = [
            [this.t.train, m.trainAcc, '#64748b'],
            [this.t.testSet, m.testAcc, '#7c3aed'],
            [this.t.worldA, m.accA, '#22707a'],
            [this.t.worldB, m.accB, '#f59e0b']
        ];
        const labelW = Math.min(150, w * 0.45);
        const rowH = h / items.length;
        ctx.textBaseline = 'middle';
        items.forEach(([label, v, colour], i) => {
            const y = i * rowH + rowH / 2;
            ctx.fillStyle = '#334155';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(label, 0, y);
            const bw = (w - labelW - 44) * v;
            ctx.fillStyle = '#e2e8f0';
            ctx.fillRect(labelW, y - 8, w - labelW - 44, 16);
            ctx.fillStyle = colour;
            ctx.fillRect(labelW, y - 8, bw, 16);
            ctx.fillStyle = '#0f172a';
            ctx.font = '600 12px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${Math.round(v * 100)}%`, w, y);
        });
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
DataBiasSim.STRINGS = [
    'db_title', 'db_badge', 'db_story', 'db_data', 'db_n', 'db_share', 'db_test', 'db_model', 'db_k', 'db_resample',
    'db_metrics', 'db_train', 'db_test_set', 'db_world_a', 'db_world_b', 'db_counts', 'db_gap', 'db_legend',
    'db_populations_shifted', 'db_populations_similar', 'db_advice_start', 'db_advice_memorize', 'db_advice_bias',
    'db_advice_small_test', 'db_advice_fair', 'db_factory_a', 'db_factory_b'
];
