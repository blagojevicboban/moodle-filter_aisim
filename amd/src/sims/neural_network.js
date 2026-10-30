/**
 * Perceptron & Neural Network: interactive training of a small multilayer network on 2D data.
 *
 * Data and initial weights come from a seeded generator, so the same settings always train to the same result
 * (a quiz review re-trains to the saved number of epochs and shows the student's own boundary).
 *
 * @module filter_aisim/sims/neural_network
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

const ACTIVATIONS = {
    sigmoid: {
        f: (z) => 1 / (1 + Math.exp(-z)),
        df: (a) => a * (1 - a)
    },
    tanh: {
        f: (z) => Math.tanh(z),
        df: (a) => 1 - a * a
    },
    relu: {
        f: (z) => (z > 0 ? z : 0),
        df: (a) => (a > 0 ? 1 : 0)
    }
};

const MAX_EPOCHS = 5000;

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

export class NeuralNetworkSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('nn_title'),
            badge: this.str('nn_badge'),
            architecture: this.str('nn_architecture'),
            layers: this.str('nn_layers'),
            layers0: this.str('nn_layers0'),
            neurons: this.str('nn_neurons'),
            activation: this.str('nn_activation'),
            training: this.str('nn_training'),
            lr: this.str('nn_lr'),
            train: this.str('nn_train'),
            pause: this.str('nn_pause'),
            step: this.str('nn_step'),
            reset: this.str('nn_reset'),
            metrics: this.str('nn_metrics'),
            epoch: this.str('nn_epoch'),
            loss: this.str('nn_loss'),
            accuracy: this.str('nn_accuracy'),
            neuronsTotal: this.str('nn_neurons_total'),
            network: this.str('nn_network'),
            lossCurve: this.str('nn_loss_curve'),
            datasets: {
                linear: this.str('nn_datasets_linear'),
                xor: this.str('nn_datasets_xor'),
                circles: this.str('nn_datasets_circles')
            },
            adviceStart: this.str('nn_advice_start'),
            advicePerceptron: this.str('nn_advice_perceptron'),
            adviceSolved: this.str('nn_advice_solved'),
            adviceDiverged: this.str('nn_advice_diverged'),
            adviceStuck: this.str('nn_advice_stuck')
        };

        this.datasetKey = ['linear', 'xor', 'circles'].includes(options.dataset) ? options.dataset : 'xor';
        this.layers = Math.min(2, Math.max(0, parseInt(options.layers, 10) || 0));
        this.neurons = Math.min(8, Math.max(1, parseInt(options.neurons, 10) || 4));
        this.activation = ACTIVATIONS[options.activation] ? options.activation : 'tanh';
        this.lr = parseFloat(options.lr) || 0.3;
        this.seed = parseInt(options.seed, 10) || 1;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;
        this.running = false;
        this.lastNotify = 0;

        this.initData();
        this.initNetwork();
        this.renderLayout();
        this.bindEvents();
        // Re-train to a saved state (quiz review, shared links): same seed and settings give the same network.
        const epochs = Math.min(MAX_EPOCHS, Math.max(0, parseInt(options.epochs, 10) || 0));
        this.trainEpochs(epochs);
        this.update(true);
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

    initData() {
        const rand = seeded(1000 + ['linear', 'xor', 'circles'].indexOf(this.datasetKey));
        const gauss = () => {
            const u = Math.max(rand(), 1e-9);
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
        };
        this.points = [];
        if (this.datasetKey === 'linear') {
            for (let i = 0; i < 24; i++) {
                this.points.push({x: -0.8 + gauss() * 0.4, y: -0.6 + gauss() * 0.4, cls: 0});
                this.points.push({x: 0.8 + gauss() * 0.4, y: 0.6 + gauss() * 0.4, cls: 1});
            }
        } else if (this.datasetKey === 'xor') {
            for (const [cx, cy] of [[-1, -1], [1, 1], [-1, 1], [1, -1]]) {
                const cls = cx * cy > 0 ? 0 : 1;
                for (let i = 0; i < 12; i++) {
                    this.points.push({x: cx + gauss() * 0.3, y: cy + gauss() * 0.3, cls});
                }
            }
        } else {
            for (let i = 0; i < 30; i++) {
                const a = 2 * Math.PI * rand();
                const r = 1.45 + gauss() * 0.12;
                this.points.push({x: Math.cos(a) * r, y: Math.sin(a) * r, cls: 0});
            }
            for (let i = 0; i < 20; i++) {
                const a = 2 * Math.PI * rand();
                const r = 0.55 * Math.sqrt(rand());
                this.points.push({x: Math.cos(a) * r, y: Math.sin(a) * r, cls: 1});
            }
        }
    }

    initNetwork() {
        const rand = seeded(this.seed * 7919);
        this.sizes = [2];
        for (let l = 0; l < this.layers; l++) {
            this.sizes.push(this.neurons);
        }
        this.sizes.push(1);
        this.W = [];
        this.b = [];
        for (let l = 1; l < this.sizes.length; l++) {
            const nIn = this.sizes[l - 1];
            const nOut = this.sizes[l];
            const limit = Math.sqrt(6 / (nIn + nOut));
            this.W.push(Array.from({length: nOut}, () => Array.from({length: nIn}, () => (rand() * 2 - 1) * limit)));
            this.b.push(Array.from({length: nOut}, () => 0));
        }
        this.epoch = 0;
        this.history = [];
    }

    /**
     * Forward pass. Hidden layers use the chosen activation, the output neuron a sigmoid.
     *
     * @param {Number} x
     * @param {Number} y
     * @returns {Array} activations of every layer (the last one holds the probability of class 1)
     */
    forward(x, y) {
        const acts = [[x, y]];
        const act = ACTIVATIONS[this.activation];
        for (let l = 0; l < this.W.length; l++) {
            const last = l === this.W.length - 1;
            const prev = acts[l];
            acts.push(this.W[l].map((row, j) => {
                let z = this.b[l][j];
                for (let i = 0; i < row.length; i++) {
                    z += row[i] * prev[i];
                }
                return last ? ACTIVATIONS.sigmoid.f(z) : act.f(z);
            }));
        }
        return acts;
    }

    predict(x, y) {
        const acts = this.forward(x, y);
        return acts[acts.length - 1][0];
    }

    /**
     * One epoch of full-batch gradient descent on the binary cross-entropy.
     */
    trainEpoch() {
        const act = ACTIVATIONS[this.activation];
        const gW = this.W.map((m) => m.map((row) => row.map(() => 0)));
        const gb = this.b.map((v) => v.map(() => 0));
        for (const pt of this.points) {
            const acts = this.forward(pt.x, pt.y);
            // Output delta for sigmoid + cross-entropy.
            let delta = [acts[acts.length - 1][0] - pt.cls];
            for (let l = this.W.length - 1; l >= 0; l--) {
                const prev = acts[l];
                for (let j = 0; j < delta.length; j++) {
                    gb[l][j] += delta[j];
                    for (let i = 0; i < prev.length; i++) {
                        gW[l][j][i] += delta[j] * prev[i];
                    }
                }
                if (l > 0) {
                    delta = this.backDelta(l, delta, prev, act);
                }
            }
        }
        const n = this.points.length;
        for (let l = 0; l < this.W.length; l++) {
            for (let j = 0; j < this.W[l].length; j++) {
                this.b[l][j] -= this.lr * gb[l][j] / n;
                for (let i = 0; i < this.W[l][j].length; i++) {
                    this.W[l][j][i] -= this.lr * gW[l][j][i] / n;
                }
            }
        }
        this.epoch++;
    }

    /**
     * Error of layer l-1, from the error of layer l (backpropagation through the weights and the activation).
     *
     * @param {Number} l layer index in W
     * @param {Array} delta error of layer l
     * @param {Array} prev activations of layer l-1
     * @param {Object} act activation of the hidden layers
     * @returns {Array}
     */
    backDelta(l, delta, prev, act) {
        return prev.map((a, i) => {
            let s = 0;
            for (let j = 0; j < delta.length; j++) {
                s += this.W[l][j][i] * delta[j];
            }
            return s * act.df(a);
        });
    }

    trainEpochs(count) {
        for (let i = 0; i < count && this.epoch < MAX_EPOCHS; i++) {
            this.trainEpoch();
            if (this.epoch % 5 === 0) {
                this.history.push(this.evaluate().loss);
            }
        }
    }

    evaluate() {
        let loss = 0;
        let correct = 0;
        for (const pt of this.points) {
            const p = Math.min(1 - 1e-7, Math.max(1e-7, this.predict(pt.x, pt.y)));
            loss -= pt.cls === 1 ? Math.log(p) : Math.log(1 - p);
            if ((p >= 0.5 ? 1 : 0) === pt.cls) {
                correct++;
            }
        }
        const n = this.points.length || 1;
        return {loss: loss / n, accuracy: correct / n};
    }

    renderLayout() {
        const opt = (value, label, current) => `<option value="${value}" ${value === current ? 'selected' : ''}>${label}</option>`;
        this.container.innerHTML = `
            <div class="aisim-container aisim-nn">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <select class="aisim-select" id="aisim-nn-dataset">
                        ${opt('linear', this.t.datasets.linear, this.datasetKey)}
                        ${opt('xor', this.t.datasets.xor, this.datasetKey)}
                        ${opt('circles', this.t.datasets.circles, this.datasetKey)}
                    </select>
                </div>

                <div class="aisim-body">
                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box" style="padding:0; overflow:hidden;">
                                <canvas class="aisim-canvas" id="aisim-nn-canvas" style="height:320px;"></canvas>
                            </div>
                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-nn-advice">${this.t.adviceStart}</div>
                            </div>
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.network}</div>
                                <canvas class="aisim-canvas" id="aisim-nn-graph" style="height:170px;"></canvas>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.architecture}</div>
                                <div class="aisim-slider-group">
                                    <label>${this.t.layers}</label>
                                    <select class="aisim-select" id="aisim-nn-layers" style="width:100%;">
                                        ${opt('0', this.t.layers0, String(this.layers))}
                                        ${opt('1', '1', String(this.layers))}
                                        ${opt('2', '2', String(this.layers))}
                                    </select>
                                </div>
                                <div class="aisim-slider-group" id="aisim-nn-neurons-group">
                                    <label>${this.t.neurons} = <span id="aisim-nn-neurons-val">${this.neurons}</span></label>
                                    <input type="range" class="aisim-slider" id="aisim-nn-neurons"
                                           min="1" max="8" step="1" value="${this.neurons}">
                                </div>
                                <div class="aisim-slider-group">
                                    <label>${this.t.activation}</label>
                                    <select class="aisim-select" id="aisim-nn-activation" style="width:100%;">
                                        ${opt('sigmoid', 'sigmoid', this.activation)}
                                        ${opt('tanh', 'tanh', this.activation)}
                                        ${opt('relu', 'ReLU', this.activation)}
                                    </select>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.training}</div>
                                <div class="aisim-slider-group">
                                    <label>${this.t.lr} α = <span id="aisim-nn-lr-val">${this.lr.toFixed(2)}</span></label>
                                    <input type="range" class="aisim-slider" id="aisim-nn-lr"
                                           min="0.01" max="3" step="0.01" value="${this.lr}">
                                </div>
                                <div class="aisim-controls">
                                    <button class="aisim-btn aisim-btn-primary" id="aisim-nn-run">${this.t.train}</button>
                                    <button class="aisim-btn aisim-btn-secondary" id="aisim-nn-step">${this.t.step}</button>
                                    <button class="aisim-btn aisim-btn-danger" id="aisim-nn-reset">${this.t.reset}</button>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.metrics}</div>
                                <div class="aisim-stat-row"><span>${this.t.epoch}</span>
                                    <span class="aisim-stat-val" id="aisim-nn-epoch">0</span></div>
                                <div class="aisim-stat-row"><span>${this.t.loss}</span>
                                    <span class="aisim-stat-val" id="aisim-nn-loss">–</span></div>
                                <div class="aisim-stat-row"><span>${this.t.accuracy}</span>
                                    <span class="aisim-stat-val" id="aisim-nn-acc">–</span></div>
                                <div class="aisim-stat-row"><span>${this.t.neuronsTotal}</span>
                                    <span class="aisim-stat-val" id="aisim-nn-hidden">0</span></div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.lossCurve}</div>
                                <canvas class="aisim-canvas" id="aisim-nn-curve" style="height:110px;"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.canvas = this.container.querySelector('#aisim-nn-canvas');
        this.graph = this.container.querySelector('#aisim-nn-graph');
        this.curve = this.container.querySelector('#aisim-nn-curve');
        this.setupDPI();
        this.container.querySelector('#aisim-nn-neurons-group').style.display = this.layers ? 'block' : 'none';
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv, fallbackH] of [['main', this.canvas, 320], ['graph', this.graph, 170], ['curve', this.curve, 110]]) {
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
        const controls = ['#aisim-nn-dataset', '#aisim-nn-layers', '#aisim-nn-neurons', '#aisim-nn-activation',
            '#aisim-nn-lr', '#aisim-nn-run', '#aisim-nn-step', '#aisim-nn-reset'];
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        // Any change of data or architecture starts from fresh weights.
        const rebuild = () => {
            this.stop();
            this.initNetwork();
            this.update(true);
        };
        q('#aisim-nn-dataset').addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.initData();
            rebuild();
        });
        q('#aisim-nn-layers').addEventListener('change', (e) => {
            this.layers = parseInt(e.target.value, 10);
            q('#aisim-nn-neurons-group').style.display = this.layers ? 'block' : 'none';
            rebuild();
        });
        q('#aisim-nn-neurons').addEventListener('input', (e) => {
            this.neurons = parseInt(e.target.value, 10);
            q('#aisim-nn-neurons-val').textContent = this.neurons;
            rebuild();
        });
        q('#aisim-nn-activation').addEventListener('change', (e) => {
            this.activation = e.target.value;
            rebuild();
        });
        q('#aisim-nn-lr').addEventListener('input', (e) => {
            this.lr = parseFloat(e.target.value);
            q('#aisim-nn-lr-val').textContent = this.lr.toFixed(2);
        });
        q('#aisim-nn-run').addEventListener('click', () => (this.running ? this.stop() : this.run()));
        q('#aisim-nn-step').addEventListener('click', () => {
            this.stop();
            this.trainEpochs(10);
            this.update(true);
        });
        q('#aisim-nn-reset').addEventListener('click', () => {
            this.seed++;
            rebuild();
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
    }

    run() {
        this.running = true;
        this.container.querySelector('#aisim-nn-run').textContent = this.t.pause;
        const frame = () => {
            if (!this.running) {
                return;
            }
            this.trainEpochs(10);
            const {loss} = this.evaluate();
            this.update(false);
            if (this.epoch >= MAX_EPOCHS || !isFinite(loss)) {
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
        this.container.querySelector('#aisim-nn-run').textContent = this.t.train;
        this.update(true);
    }

    hiddenNeurons() {
        return this.layers * this.neurons;
    }

    /**
     * Refresh numbers and drawings; tell the listener (quiz answer) at most every 250 ms while training.
     *
     * @param {Boolean} notify force the onChange call
     */
    update(notify) {
        const {loss, accuracy} = this.evaluate();
        const q = (id) => this.container.querySelector(id);
        q('#aisim-nn-epoch').textContent = this.epoch;
        q('#aisim-nn-loss').textContent = isFinite(loss) ? loss.toFixed(4) : '∞';
        q('#aisim-nn-acc').textContent = `${Math.round(accuracy * 100)}%`;
        q('#aisim-nn-hidden').textContent = this.hiddenNeurons();

        let advice = this.t.adviceStart;
        if (!isFinite(loss) || loss > 5) {
            advice = this.t.adviceDiverged;
        } else if (accuracy >= 0.95) {
            advice = this.t.adviceSolved;
        } else if (this.layers === 0 && this.datasetKey !== 'linear' && this.epoch > 0) {
            advice = this.t.advicePerceptron;
        } else if (this.epoch >= 500) {
            advice = this.t.adviceStuck;
        }
        q('#aisim-nn-advice').textContent = advice;

        this.draw();

        const now = Date.now();
        if (typeof this.onChange === 'function' && (notify || now - this.lastNotify > 250)) {
            this.lastNotify = now;
            this.onChange({
                simtype: 'neural_network',
                parameters: {
                    dataset: this.datasetKey,
                    layers: this.layers,
                    neurons: this.neurons,
                    hidden: this.hiddenNeurons(),
                    activation: this.activation,
                    lr: this.lr,
                    seed: this.seed,
                    epochs: this.epoch
                },
                metrics: {
                    accuracy: Math.round(accuracy * 1000) / 1000,
                    loss: isFinite(loss) ? Math.round(loss * 10000) / 10000 : 999
                }
            });
        }
    }

    draw() {
        this.drawBoundary();
        this.drawGraph();
        this.drawCurve();
    }

    drawBoundary() {
        const {w, h, ctx} = this.sizesPx.main;
        const toPx = (x, y) => ({px: ((x + 2.2) / 4.4) * w, py: h - ((y + 2.2) / 4.4) * h});
        ctx.clearRect(0, 0, w, h);
        const step = 8;
        for (let px = 0; px < w; px += step) {
            const x = -2.2 + ((px + step / 2) / w) * 4.4;
            for (let py = 0; py < h; py += step) {
                const y = 2.2 - ((py + step / 2) / h) * 4.4;
                const p = this.predict(x, y);
                const strength = Math.min(1, Math.abs(p - 0.5) * 2) * 0.32 + 0.04;
                ctx.fillStyle = p >= 0.5 ? `rgba(239, 68, 68, ${strength})` : `rgba(37, 99, 235, ${strength})`;
                ctx.fillRect(px, py, step, step);
            }
        }
        const o = toPx(0, 0);
        ctx.strokeStyle = 'rgba(203, 213, 225, 0.7)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, o.py);
        ctx.lineTo(w, o.py);
        ctx.moveTo(o.px, 0);
        ctx.lineTo(o.px, h);
        ctx.stroke();
        for (const pt of this.points) {
            const {px, py} = toPx(pt.x, pt.y);
            ctx.fillStyle = pt.cls === 0 ? '#2563eb' : '#ef4444';
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.8;
            ctx.stroke();
        }
    }

    drawGraph() {
        const {w, h, ctx} = this.sizesPx.graph;
        ctx.clearRect(0, 0, w, h);
        const cols = this.sizes.length;
        const pos = this.sizes.map((n, l) => Array.from({length: n}, (_, i) => ({
            x: 24 + (l * (w - 48)) / Math.max(1, cols - 1),
            y: h / 2 + (i - (n - 1) / 2) * Math.min(20, (h - 24) / Math.max(1, n - 1))
        })));
        const maxW = Math.max(0.0001, ...this.W.flat(2).map((v) => Math.abs(v)));
        for (let l = 0; l < this.W.length; l++) {
            for (let j = 0; j < this.W[l].length; j++) {
                for (let i = 0; i < this.W[l][j].length; i++) {
                    const v = this.W[l][j][i];
                    ctx.strokeStyle = v >= 0 ? 'rgba(37, 99, 235, 0.75)' : 'rgba(239, 68, 68, 0.75)';
                    ctx.lineWidth = 0.5 + 3.5 * Math.abs(v) / maxW;
                    ctx.beginPath();
                    ctx.moveTo(pos[l][i].x, pos[l][i].y);
                    ctx.lineTo(pos[l + 1][j].x, pos[l + 1][j].y);
                    ctx.stroke();
                }
            }
        }
        pos.forEach((layer, l) => layer.forEach((p) => {
            let colour = '#22707a';
            if (l === 0) {
                colour = '#64748b';
            } else if (l === cols - 1) {
                colour = '#0f172a';
            }
            ctx.fillStyle = colour;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }));
    }

    drawCurve() {
        const {w, h, ctx} = this.sizesPx.curve;
        ctx.clearRect(0, 0, w, h);
        const data = this.history.filter((v) => isFinite(v));
        if (data.length < 2) {
            return;
        }
        const max = Math.max(...data, 0.1);
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2;
        ctx.beginPath();
        data.forEach((v, i) => {
            const x = 6 + (i / (data.length - 1)) * (w - 12);
            const y = h - 8 - (Math.min(v, max) / max) * (h - 16);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
NeuralNetworkSim.STRINGS = [
    'nn_title', 'nn_badge', 'nn_architecture', 'nn_layers', 'nn_layers0', 'nn_neurons', 'nn_activation', 'nn_training',
    'nn_lr', 'nn_train', 'nn_pause', 'nn_step', 'nn_reset', 'nn_metrics', 'nn_epoch', 'nn_loss', 'nn_accuracy',
    'nn_neurons_total', 'nn_network', 'nn_loss_curve', 'nn_datasets_linear', 'nn_datasets_xor', 'nn_datasets_circles',
    'nn_advice_start', 'nn_advice_perceptron', 'nn_advice_solved', 'nn_advice_diverged', 'nn_advice_stuck'
];
