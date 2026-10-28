/**
 * K-Means Clustering interactive simulation engine.
 *
 * @module filter_aisim/sims/kmeans
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

/**
 * Find the center closest to a point (squared Euclidean distance).
 *
 * @param {Object} p Point with x and y
 * @param {Array} centers Centers with x and y
 * @returns {Object} {index, dist} of the closest center
 */
const nearestCenter = (p, centers) => {
    let index = 0;
    let dist = Infinity;
    centers.forEach((c, j) => {
        const d = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
        if (d < dist) {
            dist = d;
            index = j;
        }
    });
    return {index, dist};
};

export class KMeansSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.lang = options.lang || (document.documentElement && document.documentElement.lang) || 'en';
        this.isSr = this.lang.startsWith('sr');

        this.t = {
            title: this.tr('K-sredine (K-Means klasterovanje)', 'K-Means Clustering Lab'),
            badge: this.tr('Poglavlje 4.5', 'Chapter 4.5'),
            kLabel: this.tr('Broj klastera (K):', 'Clusters (K):'),
            datasetLabel: this.tr('Skup podataka:', 'Dataset:'),
            stepBtn: this.tr('⏭ Sledeći korak', '⏭ Next Step'),
            runBtn: this.tr('▶ Pokreni', '▶ Run'),
            pauseBtn: this.tr('⏸ Pauza', '⏸ Pause'),
            reseedBtn: this.tr('🎲 Novi početni centroidi', '🎲 Random Centroids'),
            resetBtn: this.tr('↺ Reset', '↺ Reset'),
            iteration: this.tr('Korak / Faza:', 'Iteration / Phase:'),
            phaseAssign: this.tr('1. Dodela tačaka centroidima', '1. Assign points to centroids'),
            phaseUpdate: this.tr('2. Pomeranje centroida u prosek', '2. Move centroids to mean'),
            convergedMsg: this.tr('✓ Konvergirano! Centroidi su stabilni.', '✓ Converged! Centroids stable.'),
            inertiaLabel: this.tr('Inercija (WCSS):', 'Inertia (WCSS):'),
            elbowTitle: this.tr('Metoda lakta (Elbow Method)', 'Elbow Plot (WCSS vs K)'),
            addPointNotice: this.tr(
                'Nenadgledano učenje: tačke nemaju labele. Kliknite na platno da dodate tačku.',
                'Unsupervised: points have no labels. Click canvas to add custom points.'
            ),
            adviceLowK: this.tr(
                'Premalo K: Prirodne grupe su spojene u jedan preveliki klaster.',
                'K is too small: Natural groups are merged together.'
            ),
            adviceElbow: this.tr(
                'Optimalno K: Na "prelomu lakta" inercija naglo usporava pad.',
                'Optimal K: The "elbow" marks the point of diminishing returns.'
            ),
            adviceHighK: this.tr(
                'Preveliko K: Prirodne celine se veštački seckaju na sitne klastere.',
                'K is too large: Natural clusters are fragmented artificially.'
            ),
            datasets: {
                blobs3: this.tr('🫧 3 prirodna klastera', '🫧 3 Natural Clusters'),
                blobs4: this.tr('🧭 4 industrijska senzorska režima', '🧭 4 Sensor Regimes'),
                uniform: this.tr('🎲 Ravnomerno raspoređene tačke', '🎲 Uniform Random Points')
            }
        };

        this.k = parseInt(options.k, 10) || 3;
        this.datasetKey = options.dataset || 'blobs3';
        this.iteration = 0;
        this.phase = 'assign'; // 'assign' or 'update'
        this.isRunning = false;
        this.timer = null;
        this.isConverged = false;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.colors = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2'];

        this.initData();
        this.renderLayout();
        this.bindEvents();
        this.update();
    }

    /**
     * Pick the Serbian or English variant of a UI text.
     *
     * @param {String} sr Serbian text
     * @param {String} en English text
     * @returns {String}
     */
    tr(sr, en) {
        return this.isSr ? sr : en;
    }

    initData() {
        this.points = [];
        this.centroids = [];
        this.iteration = 0;
        this.phase = 'assign';
        this.isConverged = false;

        let seed = 77;
        const rand = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return (seed / 233280) * 2 - 1;
        };

        if (this.datasetKey === 'blobs3') {
            const centers = [
                {x: -1.0, y: 0.9},
                {x: 1.0, y: 0.8},
                {x: 0.0, y: -1.0}
            ];
            for (const c of centers) {
                for (let i = 0; i < 22; i++) {
                    this.points.push({
                        x: c.x + rand() * 0.45,
                        y: c.y + rand() * 0.45,
                        cluster: -1
                    });
                }
            }
        } else if (this.datasetKey === 'blobs4') {
            const centers = [
                {x: -1.1, y: 1.0},
                {x: 1.1, y: 1.0},
                {x: -1.0, y: -0.9},
                {x: 1.0, y: -0.9}
            ];
            for (const c of centers) {
                for (let i = 0; i < 16; i++) {
                    this.points.push({
                        x: c.x + rand() * 0.4,
                        y: c.y + rand() * 0.4,
                        cluster: -1
                    });
                }
            }
        } else {
            for (let i = 0; i < 65; i++) {
                this.points.push({
                    x: rand() * 1.6,
                    y: rand() * 1.6,
                    cluster: -1
                });
            }
        }

        this.initCentroids();
        this.computeElbowPoints();
    }

    initCentroids() {
        this.centroids = [];
        for (let i = 0; i < this.k; i++) {
            // Pick random existing points as initial centroids.
            const idx = Math.floor(Math.random() * this.points.length);
            const p = this.points[idx] || {x: (Math.random() - 0.5) * 2, y: (Math.random() - 0.5) * 2};
            this.centroids.push({x: p.x, y: p.y, prevX: p.x, prevY: p.y});
        }
        this.isConverged = false;
        this.iteration = 0;
        this.phase = 'assign';
    }

    renderLayout() {
        const isB3 = this.datasetKey === 'blobs3' ? 'selected' : '';
        const isB4 = this.datasetKey === 'blobs4' ? 'selected' : '';
        const isUn = this.datasetKey === 'uniform' ? 'selected' : '';

        this.container.innerHTML = `
            <div class="aisim-container aisim-kmeans">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <div>
                        <select class="aisim-select" id="aisim-km-dataset">
                            <option value="blobs3" ${isB3}>${this.t.datasets.blobs3}</option>
                            <option value="blobs4" ${isB4}>${this.t.datasets.blobs4}</option>
                            <option value="uniform" ${isUn}>${this.t.datasets.uniform}</option>
                        </select>
                    </div>
                </div>

                <div class="aisim-body">
                    <div class="aisim-controls">
                        <button class="aisim-btn aisim-btn-primary" id="aisim-km-btn-run">${this.t.runBtn}</button>
                        <button class="aisim-btn aisim-btn-secondary" id="aisim-km-btn-step">${this.t.stepBtn}</button>
                        <button class="aisim-btn aisim-btn-secondary" id="aisim-km-btn-reseed">${this.t.reseedBtn}</button>
                        <button class="aisim-btn aisim-btn-danger" id="aisim-km-btn-reset">${this.t.resetBtn}</button>
                    </div>

                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box" style="padding:0; overflow:hidden;">
                                <canvas class="aisim-canvas" id="aisim-km-canvas"
                                        style="height:320px; cursor:crosshair;"></canvas>
                            </div>

                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-km-advice">
                                    ${this.t.addPointNotice}
                                </div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.kLabel}</div>
                                <div class="aisim-slider-group">
                                    <label>K = <span id="val-km-k">${this.k}</span></label>
                                    <input type="range" class="aisim-slider" id="slider-km-k"
                                           min="2" max="6" step="1" value="${this.k}">
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">Stanje algoritma</div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.iteration}</span>
                                    <span class="aisim-stat-val" id="val-km-iter">0</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>Faza:</span>
                                    <span class="aisim-stat-val" id="val-km-phase" style="font-size:0.8rem;">
                                        ${this.t.phaseAssign}
                                    </span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.inertiaLabel}</span>
                                    <span class="aisim-stat-val" id="val-km-inertia">0.00</span>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.elbowTitle}</div>
                                <canvas class="aisim-mini-chart" id="aisim-km-elbow-canvas"></canvas>
                                <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                                            color:#64748b; margin-top:4px;">
                                    <span>K = 1</span>
                                    <span>Inercija (WCSS)</span>
                                    <span>K = 6</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="aisim-status-bar" id="aisim-km-status">
                        ${this.t.addPointNotice}
                    </div>
                </div>
            </div>
        `;

        this.canvas = this.container.querySelector('#aisim-km-canvas');
        this.elbowCanvas = this.container.querySelector('#aisim-km-elbow-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.elbowCtx = this.elbowCanvas.getContext('2d');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = (rect.width || 540) * dpr;
        this.canvas.height = (rect.height || 320) * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvasWidth = rect.width || 540;
        this.canvasHeight = rect.height || 320;

        const eRect = this.elbowCanvas.getBoundingClientRect();
        this.elbowCanvas.width = (eRect.width || 280) * dpr;
        this.elbowCanvas.height = (eRect.height || 120) * dpr;
        this.elbowCtx.scale(dpr, dpr);
        this.elbowWidth = eRect.width || 280;
        this.elbowHeight = eRect.height || 120;
    }

    bindEvents() {
        const btnRun = this.container.querySelector('#aisim-km-btn-run');
        const btnStep = this.container.querySelector('#aisim-km-btn-step');
        const btnReseed = this.container.querySelector('#aisim-km-btn-reseed');
        const btnReset = this.container.querySelector('#aisim-km-btn-reset');
        const sliderK = this.container.querySelector('#slider-km-k');
        const selDataset = this.container.querySelector('#aisim-km-dataset');

        if (this.readonly) {
            if (btnRun) {
                btnRun.disabled = true;
            }
            if (btnStep) {
                btnStep.disabled = true;
            }
            if (btnReseed) {
                btnReseed.disabled = true;
            }
            if (btnReset) {
                btnReset.disabled = true;
            }
            if (sliderK) {
                sliderK.disabled = true;
            }
            if (selDataset) {
                selDataset.disabled = true;
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

        btnReseed.addEventListener('click', () => {
            this.pause();
            this.initCentroids();
            this.update();
        });

        btnReset.addEventListener('click', () => {
            this.pause();
            this.initData();
            this.update();
        });

        sliderK.addEventListener('input', (e) => {
            this.pause();
            this.k = parseInt(e.target.value, 10);
            this.container.querySelector('#val-km-k').textContent = this.k;
            this.initCentroids();
            this.update();
        });

        selDataset.addEventListener('change', (e) => {
            this.pause();
            this.datasetKey = e.target.value;
            this.initData();
            this.update();
        });

        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const x = -2.0 + (clickX / this.canvasWidth) * 4.0;
            const y = 2.0 - (clickY / this.canvasHeight) * 4.0;
            this.points.push({x, y, cluster: -1});
            this.computeElbowPoints();
            this.update();
        });

        window.addEventListener('resize', () => {
            this.setupDPI();
            this.update();
        });
    }

    toPx(x, y) {
        const px = ((x - (-2.0)) / 4.0) * this.canvasWidth;
        const py = this.canvasHeight - ((y - (-2.0)) / 4.0) * this.canvasHeight;
        return {px, py};
    }

    step() {
        if (this.isConverged) {
            this.pause();
            return;
        }

        if (this.phase === 'assign') {
            // Assign each point to closest centroid.
            for (const pt of this.points) {
                pt.cluster = nearestCenter(pt, this.centroids).index;
            }

            this.phase = 'update';
            this.container.querySelector('#val-km-phase').textContent = this.t.phaseUpdate;
        } else {
            // Update step: move centroids to cluster mean.
            let maxShift = 0;
            for (let j = 0; j < this.centroids.length; j++) {
                const c = this.centroids[j];
                const members = this.points.filter(p => p.cluster === j);
                if (members.length > 0) {
                    const sumX = members.reduce((acc, p) => acc + p.x, 0);
                    const sumY = members.reduce((acc, p) => acc + p.y, 0);
                    const newX = sumX / members.length;
                    const newY = sumY / members.length;
                    const shift = Math.hypot(newX - c.x, newY - c.y);
                    if (shift > maxShift) {
                        maxShift = shift;
                    }
                    c.prevX = c.x;
                    c.prevY = c.y;
                    c.x = newX;
                    c.y = newY;
                }
            }

            this.iteration++;
            this.container.querySelector('#val-km-iter').textContent = this.iteration;

            if (maxShift < 0.002) {
                this.isConverged = true;
                this.pause();
                this.container.querySelector('#val-km-phase').textContent = this.t.convergedMsg;
                const statBar = this.container.querySelector('#aisim-km-status');
                statBar.className = 'aisim-status-bar converged';
                statBar.textContent = this.t.convergedMsg;
            } else {
                this.phase = 'assign';
                this.container.querySelector('#val-km-phase').textContent = this.t.phaseAssign;
            }
        }

        this.update();
    }

    run() {
        this.isRunning = true;
        this.container.querySelector('#aisim-km-btn-run').textContent = this.t.pauseBtn;
        this.timer = setInterval(() => {
            this.step();
        }, 350);
    }

    pause() {
        this.isRunning = false;
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        const btn = this.container.querySelector('#aisim-km-btn-run');
        if (btn) {
            btn.textContent = this.t.runBtn;
        }
    }

    calcInertia() {
        let inertia = 0;
        for (const pt of this.points) {
            if (pt.cluster >= 0 && pt.cluster < this.centroids.length) {
                const c = this.centroids[pt.cluster];
                inertia += Math.pow(pt.x - c.x, 2) + Math.pow(pt.y - c.y, 2);
            }
        }
        return inertia;
    }

    computeElbowPoints() {
        this.elbowCurve = [];
        // Approximate K-means clustering for K=1..6 to compute WCSS elbow curve.
        for (let kVal = 1; kVal <= 6; kVal++) {
            let centers = [];
            for (let i = 0; i < kVal; i++) {
                const pt = this.points[Math.floor((i / kVal) * this.points.length)] || {x: 0, y: 0};
                centers.push({x: pt.x, y: pt.y});
            }

            for (let iter = 0; iter < 6; iter++) {
                const groups = Array.from({length: kVal}, () => []);
                for (const p of this.points) {
                    groups[nearestCenter(p, centers).index].push(p);
                }
                for (let j = 0; j < kVal; j++) {
                    if (groups[j].length > 0) {
                        centers[j].x = groups[j].reduce((acc, p) => acc + p.x, 0) / groups[j].length;
                        centers[j].y = groups[j].reduce((acc, p) => acc + p.y, 0) / groups[j].length;
                    }
                }
            }

            let wcss = 0;
            for (const p of this.points) {
                wcss += nearestCenter(p, centers).dist;
            }
            this.elbowCurve.push({k: kVal, wcss});
        }
    }

    update() {
        const inertia = this.calcInertia();
        this.container.querySelector('#val-km-inertia').textContent = inertia.toFixed(2);

        const adviceBox = this.container.querySelector('#aisim-km-advice');
        if (this.k < 3) {
            adviceBox.textContent = this.t.adviceLowK;
        } else if (this.k === 3 || this.k === 4) {
            adviceBox.textContent = this.t.adviceElbow;
        } else {
            adviceBox.textContent = this.t.adviceHighK;
        }

        this.draw();
        this.drawElbow();

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'kmeans',
                parameters: {
                    k: this.k,
                    dataset: this.datasetKey,
                    iterations: this.iteration
                },
                metrics: {
                    wcss: Math.round(inertia * 100) / 100,
                    converged: this.isConverged
                }
            });
        }
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;
        ctx.clearRect(0, 0, w, h);

        // Draw connecting assignment lines.
        if (this.phase === 'update' || this.isConverged) {
            ctx.lineWidth = 1;
            for (const pt of this.points) {
                if (pt.cluster >= 0 && pt.cluster < this.centroids.length) {
                    const c = this.centroids[pt.cluster];
                    const pPx = this.toPx(pt.x, pt.y);
                    const cPx = this.toPx(c.x, c.y);
                    ctx.strokeStyle = this.colors[pt.cluster % this.colors.length] + '33';
                    ctx.beginPath();
                    ctx.moveTo(pPx.px, pPx.py);
                    ctx.lineTo(cPx.px, cPx.py);
                    ctx.stroke();
                }
            }
        }

        // Draw points.
        for (const pt of this.points) {
            const {px, py} = this.toPx(pt.x, pt.y);
            const color = pt.cluster >= 0 ? this.colors[pt.cluster % this.colors.length] : '#94a3b8';
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.stroke();
        }

        // Draw centroids as glowing crosshair stars.
        for (let j = 0; j < this.centroids.length; j++) {
            const c = this.centroids[j];
            const {px, py} = this.toPx(c.x, c.y);
            const color = this.colors[j % this.colors.length];

            // Outer pulse circle.
            ctx.fillStyle = color + '2a';
            ctx.beginPath();
            ctx.arc(px, py, 14, 0, Math.PI * 2);
            ctx.fill();

            // Centroid body.
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Crosshair inner mark.
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(px - 4, py);
            ctx.lineTo(px + 4, py);
            ctx.moveTo(px, py - 4);
            ctx.lineTo(px, py + 4);
            ctx.stroke();
        }
    }

    drawElbow() {
        const ctx = this.elbowCtx;
        const w = this.elbowWidth;
        const h = this.elbowHeight;
        ctx.clearRect(0, 0, w, h);

        if (!this.elbowCurve || this.elbowCurve.length === 0) {
            return;
        }

        const pad = 20;
        const gw = w - 2 * pad;
        const gh = h - 2 * pad;
        const maxWcss = Math.max(...this.elbowCurve.map(e => e.wcss), 1);

        // Draw elbow curve line.
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.elbowCurve.length; i++) {
            const item = this.elbowCurve[i];
            const px = pad + (i / (this.elbowCurve.length - 1)) * gw;
            const py = (h - pad) - (item.wcss / maxWcss) * gh;
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Draw dots on elbow points.
        for (let i = 0; i < this.elbowCurve.length; i++) {
            const item = this.elbowCurve[i];
            const px = pad + (i / (this.elbowCurve.length - 1)) * gw;
            const py = (h - pad) - (item.wcss / maxWcss) * gh;
            const isSelected = item.k === this.k;

            ctx.fillStyle = isSelected ? '#ef4444' : '#7c3aed';
            ctx.beginPath();
            ctx.arc(px, py, isSelected ? 5.5 : 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }
}
