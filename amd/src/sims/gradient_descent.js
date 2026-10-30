/**
 * Gradient Descent interactive simulation engine.
 *
 * @module filter_aisim/sims/gradient_descent
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

export class GradientDescentSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        // Translations dictionary.
        this.t = {
            title: this.str('gd_title'),
            badge: this.str('gd_badge'),
            run: this.str('gd_run'),
            pause: this.str('gd_pause'),
            step: this.str('gd_step'),
            reset: this.str('gd_reset'),
            lr: this.str('gd_lr'),
            w0: this.str('gd_w0'),
            func: this.str('gd_func'),
            convex: this.str('gd_convex'),
            nonconvex: this.str('gd_nonconvex'),
            plateau: this.str('gd_plateau'),
            epoch: this.str('gd_epoch'),
            wVal: this.str('gd_w_val'),
            costVal: this.str('gd_cost_val'),
            gradVal: this.str('gd_grad_val'),
            costHist: this.str('gd_cost_hist'),
            statusReady: this.str('gd_status_ready'),
            statusRunning: this.str('gd_status_running'),
            statusConverged: this.str('gd_status_converged'),
            statusDiverging: this.str('gd_status_diverging'),
            statusOscillating: this.str('gd_status_oscillating')
        };

        // Mathematical function profiles.
        this.profiles = {
            convex: {
                name: this.t.convex,
                wMin: -1.5,
                wMax: 5.5,
                jMin: 0,
                jMax: 12,
                f: (w) => 0.5 * Math.pow(w - 2.0, 2) + 1.0,
                df: (w) => (w - 2.0),
                defaultW0: -0.5,
                defaultAlpha: 0.15
            },
            nonconvex: {
                name: this.t.nonconvex,
                wMin: -2.8,
                wMax: 3.2,
                jMin: 0,
                jMax: 10,
                f: (w) => 0.15 * Math.pow(w, 4) - 0.2 * Math.pow(w, 3) - 1.2 * Math.pow(w, 2) + 0.5 * w + 4.5,
                df: (w) => 0.6 * Math.pow(w, 3) - 0.6 * Math.pow(w, 2) - 2.4 * w + 0.5,
                defaultW0: -2.2,
                defaultAlpha: 0.08
            },
            plateau: {
                name: this.t.plateau,
                wMin: -3.0,
                wMax: 3.0,
                jMin: 0,
                jMax: 8,
                f: (w) => 0.2 * Math.pow(w, 4) - 0.8 * Math.pow(w, 2) + 0.4 * Math.sin(3 * w) + 3.0,
                df: (w) => 0.8 * Math.pow(w, 3) - 1.6 * w + 1.2 * Math.cos(3 * w),
                defaultW0: -2.6,
                defaultAlpha: 0.05
            }
        };

        this.selectedProfileKey = options.profile || 'convex';
        this.currentProfile = this.profiles[this.selectedProfileKey] || this.profiles.convex;

        // Hyperparameters & State.
        this.alpha = parseFloat(options.lr) || this.currentProfile.defaultAlpha;
        this.w0 = parseFloat(options.w0) || this.currentProfile.defaultW0;
        this.w = this.w0;
        this.epoch = 0;
        this.history = [];
        this.isRunning = false;
        this.timer = null;
        this.speed = 100; // Speed in ms per step.
        this.status = 'ready';
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.renderLayout();
        this.bindEvents();
        this.reset();
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

    renderLayout() {
        const isConv = this.selectedProfileKey === 'convex' ? 'selected' : '';
        const isNonConv = this.selectedProfileKey === 'nonconvex' ? 'selected' : '';
        const isPlat = this.selectedProfileKey === 'plateau' ? 'selected' : '';

        this.container.innerHTML = `
            <div class="aisim-container">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <div class="aisim-profile-select">
                        <select class="aisim-select" id="aisim-prof">
                            <option value="convex" ${isConv}>${this.profiles.convex.name}</option>
                            <option value="nonconvex" ${isNonConv}>${this.profiles.nonconvex.name}</option>
                            <option value="plateau" ${isPlat}>${this.profiles.plateau.name}</option>
                        </select>
                    </div>
                </div>
                <div class="aisim-body">
                    <div class="aisim-controls">
                        <button class="aisim-btn aisim-btn-primary" id="aisim-btn-run">${this.t.run}</button>
                        <button class="aisim-btn aisim-btn-secondary" id="aisim-btn-step">${this.t.step}</button>
                        <button class="aisim-btn aisim-btn-danger" id="aisim-btn-reset">${this.t.reset}</button>
                    </div>

                    <div class="aisim-grid">
                        <div class="aisim-canvas-box">
                            <canvas class="aisim-canvas" id="aisim-main-canvas"></canvas>
                        </div>
                        <div class="aisim-sidebar">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.lr}</div>
                                <div class="aisim-slider-group">
                                    <label>α = <span id="aisim-lr-val">${this.alpha.toFixed(3)}</span></label>
                                    <input type="range" class="aisim-slider" id="aisim-lr"
                                           min="0.005" max="1.1" step="0.005" value="${this.alpha}">
                                </div>
                                <div class="aisim-slider-group">
                                    <label>${this.t.w0}: <span id="aisim-w0-val">${this.w0.toFixed(2)}</span></label>
                                    <input type="range" class="aisim-slider" id="aisim-w0"
                                           min="${this.currentProfile.wMin}" max="${this.currentProfile.wMax}"
                                           step="0.1" value="${this.w0}">
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.str('gd_metrics')}</div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.epoch}</span>
                                    <span class="aisim-stat-val" id="aisim-stat-epoch">0</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.wVal}</span>
                                    <span class="aisim-stat-val" id="aisim-stat-w">0.000</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.costVal}</span>
                                    <span class="aisim-stat-val" id="aisim-stat-cost">0.000</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.gradVal}</span>
                                    <span class="aisim-stat-val" id="aisim-stat-grad">0.000</span>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.costHist}</div>
                                <canvas class="aisim-mini-chart" id="aisim-mini-canvas"></canvas>
                            </div>
                        </div>
                    </div>

                    <div class="aisim-status-bar" id="aisim-status">
                        ${this.t.statusReady}
                    </div>
                </div>
            </div>
        `;

        this.canvas = this.container.querySelector('#aisim-main-canvas');
        this.miniCanvas = this.container.querySelector('#aisim-mini-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.miniCtx = this.miniCanvas.getContext('2d');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = (rect.width || 560) * dpr;
        this.canvas.height = (rect.height || 320) * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvasWidth = rect.width || 560;
        this.canvasHeight = rect.height || 320;

        const miniRect = this.miniCanvas.getBoundingClientRect();
        this.miniCanvas.width = (miniRect.width || 280) * dpr;
        this.miniCanvas.height = (miniRect.height || 120) * dpr;
        this.miniCtx.scale(dpr, dpr);
        this.miniWidth = miniRect.width || 280;
        this.miniHeight = miniRect.height || 120;
    }

    bindEvents() {
        const btnRun = this.container.querySelector('#aisim-btn-run');
        const btnStep = this.container.querySelector('#aisim-btn-step');
        const btnReset = this.container.querySelector('#aisim-btn-reset');
        const sliderLr = this.container.querySelector('#aisim-lr');
        const sliderW0 = this.container.querySelector('#aisim-w0');
        const selProf = this.container.querySelector('#aisim-prof');

        if (this.readonly) {
            if (btnRun) {
                btnRun.disabled = true;
            }
            if (btnStep) {
                btnStep.disabled = true;
            }
            if (btnReset) {
                btnReset.disabled = true;
            }
            if (sliderLr) {
                sliderLr.disabled = true;
            }
            if (sliderW0) {
                sliderW0.disabled = true;
            }
            if (selProf) {
                selProf.disabled = true;
            }
            return;
        }

        btnRun.addEventListener('click', () => {
            if (this.isRunning) {
                this.pause();
            } else {
                this.run();
            }
        });

        btnStep.addEventListener('click', () => {
            this.pause();
            this.step();
        });

        btnReset.addEventListener('click', () => {
            this.pause();
            this.reset();
        });

        sliderLr.addEventListener('input', (e) => {
            this.alpha = parseFloat(e.target.value);
            this.container.querySelector('#aisim-lr-val').textContent = this.alpha.toFixed(3);
        });

        sliderW0.addEventListener('input', (e) => {
            this.w0 = parseFloat(e.target.value);
            this.container.querySelector('#aisim-w0-val').textContent = this.w0.toFixed(2);
            this.reset();
        });

        selProf.addEventListener('change', (e) => {
            this.selectedProfileKey = e.target.value;
            this.currentProfile = this.profiles[this.selectedProfileKey];
            sliderW0.min = this.currentProfile.wMin;
            sliderW0.max = this.currentProfile.wMax;
            this.alpha = this.currentProfile.defaultAlpha;
            this.w0 = this.currentProfile.defaultW0;
            sliderLr.value = this.alpha;
            sliderW0.value = this.w0;
            this.container.querySelector('#aisim-lr-val').textContent = this.alpha.toFixed(3);
            this.container.querySelector('#aisim-w0-val').textContent = this.w0.toFixed(2);
            this.reset();
        });

        // Click on canvas to place initial w0.
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const wClicked = this.pixelToW(clickX);
            if (wClicked >= this.currentProfile.wMin && wClicked <= this.currentProfile.wMax) {
                this.w0 = wClicked;
                sliderW0.value = this.w0;
                this.container.querySelector('#aisim-w0-val').textContent = this.w0.toFixed(2);
                this.reset();
            }
        });

        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
    }

    wToPixel(w) {
        const {wMin, wMax} = this.currentProfile;
        const padding = 45;
        const wRange = wMax - wMin;
        return padding + ((w - wMin) / wRange) * (this.canvasWidth - 2 * padding);
    }

    pixelToW(px) {
        const {wMin, wMax} = this.currentProfile;
        const padding = 45;
        const usableWidth = this.canvasWidth - 2 * padding;
        return wMin + ((px - padding) / usableWidth) * (wMax - wMin);
    }

    jToPixel(j) {
        const {jMin, jMax} = this.currentProfile;
        const padding = 35;
        const usableHeight = this.canvasHeight - 2 * padding;
        const norm = (j - jMin) / (jMax - jMin);
        return (this.canvasHeight - padding) - norm * usableHeight;
    }

    reset() {
        this.w = this.w0;
        this.epoch = 0;
        this.history = [];
        this.status = 'ready';
        const cost = this.currentProfile.f(this.w);
        const grad = this.currentProfile.df(this.w);
        this.history.push({epoch: 0, w: this.w, cost, grad});
        this.updateStats(cost, grad);
        this.setStatus(this.t.statusReady, 'ready');
        this.draw();
    }

    step() {
        if (this.epoch >= 150) {
            this.pause();
            return;
        }

        const cost = this.currentProfile.f(this.w);
        const grad = this.currentProfile.df(this.w);

        // Check divergence.
        if (Math.abs(this.w) > 20 || cost > 150 || isNaN(cost)) {
            this.setStatus(this.t.statusDiverging, 'diverging');
            this.pause();
            return;
        }

        // Check convergence.
        if (Math.abs(grad) < 0.005 && this.epoch > 1) {
            this.setStatus(this.t.statusConverged, 'converged');
            this.pause();
            return;
        }

        // Gradient update step: w := w - alpha * grad.
        const nextW = this.w - this.alpha * grad;
        this.epoch += 1;
        this.w = nextW;

        const nextCost = this.currentProfile.f(this.w);
        const nextGrad = this.currentProfile.df(this.w);
        this.history.push({epoch: this.epoch, w: this.w, cost: nextCost, grad: nextGrad});

        this.updateStats(nextCost, nextGrad);
        this.draw();
    }

    run() {
        this.isRunning = true;
        this.container.querySelector('#aisim-btn-run').textContent = this.t.pause;
        this.setStatus(this.t.statusRunning, 'running');
        this.timer = setInterval(() => {
            this.step();
        }, this.speed);
    }

    pause() {
        this.isRunning = false;
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        const btn = this.container.querySelector('#aisim-btn-run');
        if (btn) {
            btn.textContent = this.t.run;
        }
    }

    setStatus(msg, cls) {
        const bar = this.container.querySelector('#aisim-status');
        if (bar) {
            bar.className = 'aisim-status-bar ' + cls;
            bar.textContent = msg;
        }
    }

    updateStats(cost, grad) {
        this.container.querySelector('#aisim-stat-epoch').textContent = this.epoch;
        this.container.querySelector('#aisim-stat-w').textContent = isNaN(this.w) ? 'NaN' : this.w.toFixed(4);
        this.container.querySelector('#aisim-stat-cost').textContent = isNaN(cost) ? 'Inf' : cost.toFixed(4);
        this.container.querySelector('#aisim-stat-grad').textContent = isNaN(grad) ? 'NaN' : grad.toFixed(4);

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'gradient_descent',
                parameters: {
                    w: Math.round(this.w * 1000) / 1000,
                    learningRate: this.alpha,
                    steps: this.epoch
                },
                metrics: {
                    loss: Math.round(cost * 10000) / 10000,
                    grad: Math.round(grad * 10000) / 10000
                }
            });
        }
    }

    draw() {
        this.drawMainCurve();
        this.drawMiniChart();
    }

    drawMainCurve() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;
        const prof = this.currentProfile;

        ctx.clearRect(0, 0, w, h);

        // Draw grid lines.
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for (let j = Math.floor(prof.jMin); j <= prof.jMax; j += 2) {
            const py = this.jToPixel(j);
            ctx.beginPath();
            ctx.moveTo(35, py);
            ctx.lineTo(w - 20, py);
            ctx.stroke();

            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px sans-serif';
            ctx.fillText(j.toString(), 15, py + 3);
        }

        // Draw axes.
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 1.5;
        const yZero = this.jToPixel(0);
        ctx.beginPath();
        ctx.moveTo(40, yZero);
        ctx.lineTo(w - 20, yZero);
        ctx.stroke();

        // Draw function curve J(w).
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const steps = 150;
        const dw = (prof.wMax - prof.wMin) / steps;
        for (let i = 0; i <= steps; i++) {
            const cw = prof.wMin + i * dw;
            const cj = prof.f(cw);
            const px = this.wToPixel(cw);
            const py = this.jToPixel(cj);
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Draw trajectory history.
        if (this.history.length > 1) {
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            for (let i = 0; i < this.history.length; i++) {
                const pt = this.history[i];
                const px = this.wToPixel(pt.w);
                const py = this.jToPixel(pt.cost);
                if (i === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw historical dots.
            for (let i = 0; i < this.history.length - 1; i++) {
                const pt = this.history[i];
                ctx.fillStyle = '#f59e0b';
                ctx.beginPath();
                ctx.arc(this.wToPixel(pt.w), this.jToPixel(pt.cost), 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw current point (w, J(w)).
        const curCost = prof.f(this.w);
        const curGrad = prof.df(this.w);
        const curPx = this.wToPixel(this.w);
        const curPy = this.jToPixel(curCost);

        // Tangent line: slope = curGrad.
        if (!isNaN(curGrad) && Math.abs(curGrad) < 50) {
            const tangentLength = 40;
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const dx = tangentLength;
            const dy = tangentLength * curGrad * ((h - 70) / (prof.jMax - prof.jMin)) /
                       ((w - 90) / (prof.wMax - prof.wMin));
            ctx.moveTo(curPx - dx, curPy + dy);
            ctx.lineTo(curPx + dx, curPy - dy);
            ctx.stroke();
        }

        // Draw glowing ball.
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(curPx, curPy, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    drawMiniChart() {
        const ctx = this.miniCtx;
        const w = this.miniWidth;
        const h = this.miniHeight;
        ctx.clearRect(0, 0, w, h);

        if (this.history.length < 2) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px sans-serif';
            ctx.fillText(this.str('gd_awaiting_training'), 15, h / 2);
            return;
        }

        const maxCost = Math.max(...this.history.map(pt => (isNaN(pt.cost) ? 0 : pt.cost)), 5);
        const minCost = Math.min(...this.history.map(pt => (isNaN(pt.cost) ? 0 : pt.cost)), 0);
        const costRange = Math.max(maxCost - minCost, 1);

        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.history.length; i++) {
            const pt = this.history[i];
            const px = 15 + (i / Math.max(this.history.length - 1, 10)) * (w - 30);
            const py = (h - 15) - ((pt.cost - minCost) / costRange) * (h - 30);
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
GradientDescentSim.STRINGS = [
    'gd_title', 'gd_badge', 'gd_run', 'gd_pause', 'gd_step', 'gd_reset', 'gd_lr', 'gd_w0', 'gd_func', 'gd_convex',
    'gd_nonconvex', 'gd_plateau', 'gd_epoch', 'gd_w_val', 'gd_cost_val', 'gd_grad_val', 'gd_cost_hist',
    'gd_status_ready', 'gd_status_running', 'gd_status_converged', 'gd_status_diverging', 'gd_status_oscillating',
    'gd_metrics', 'gd_awaiting_training'
];
