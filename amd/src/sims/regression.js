/**
 * Polynomial Regression & Overfitting/Underfitting interactive simulation engine.
 *
 * @module filter_aisim/sims/regression
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

export class RegressionSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.lang = options.lang || (document.documentElement && document.documentElement.lang) || 'en';
        this.isSr = this.lang.startsWith('sr');

        this.t = {
            title: this.tr('Polinomska regresija i preprilagođavanje', 'Polynomial Regression & Overfitting'),
            badge: this.tr('Poglavlja 3.3 & 3.10', 'Chapters 3.3 & 3.10'),
            degreeLabel: this.tr('Stepen polinoma (d):', 'Polynomial Degree (d):'),
            regLabel: this.tr('Regularizacija (λ):', 'Regularization (λ):'),
            datasetLabel: this.tr('Skup podataka:', 'Dataset:'),
            trainPts: this.tr('Trening tačke (Train)', 'Training Points'),
            testPts: this.tr('Validacione tačke (Validation)', 'Validation Points'),
            polyCurve: this.tr('Polinom ŷ(x)', 'Polynomial ŷ(x)'),
            trainError: this.tr('Trening MSE greška:', 'Training MSE:'),
            testError: this.tr('Validacioni MSE:', 'Validation MSE:'),
            complexityPlotTitle: this.tr('Složenost vs Greška (U-kriva)', 'Model Complexity vs Error (U-Curve)'),
            underfitting: this.tr(
                'Potprilagođavanje (Underfitting): Model je previše prost (High Bias).',
                'Underfitting: Model is too simple (High Bias).'
            ),
            optimal: this.tr(
                'Optimalno prilagođavanje (Good Fit): Dobra generalizacija na nove podatke.',
                'Good Fit: Optimal generalization to unseen data.'
            ),
            overfitting: this.tr(
                'Preprilagođavanje (Overfitting): Model je naučio šum umesto pravila (High Variance)!',
                'Overfitting: Model memorized noise instead of signal (High Variance)!'
            ),
            datasets: {
                sine: this.tr('🌊 Sinusni talas sa šumom', '🌊 Sine Wave with Noise'),
                quadratic: this.tr('📈 Kvadratni trend', '📈 Quadratic Trend'),
                engine: this.tr('⚙️ Potrošnja motora po obrtajima', '⚙️ Engine RPM vs Fuel Consumption')
            }
        };

        this.degree = parseInt(options.degree, 10) || 3;
        this.lambda = parseFloat(options.lambda) || 0.0;
        this.datasetKey = options.dataset || 'sine';
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

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
        this.trainPoints = [];
        this.testPoints = [];

        const numTrain = 14;
        const numTest = 14;

        const fTrue = (x) => {
            if (this.datasetKey === 'sine') {
                return Math.sin(x * Math.PI) * 1.5;
            } else if (this.datasetKey === 'quadratic') {
                return 1.2 * Math.pow(x, 2) - 0.8 * x - 0.8;
            } else {
                return 0.8 * Math.pow(x + 0.2, 3) - 1.2 * x + 0.2;
            }
        };

        let seed = 42;
        const rand = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return (seed / 233280) * 2 - 1;
        };

        for (let i = 0; i < numTrain; i++) {
            const x = -1.6 + (3.2 / (numTrain - 1)) * i;
            const noise = rand() * 0.45;
            this.trainPoints.push({x, y: fTrue(x) + noise});
        }

        for (let i = 0; i < numTest; i++) {
            const x = -1.5 + (3.0 / (numTest - 1)) * i + (rand() * 0.08);
            const noise = rand() * 0.45;
            this.testPoints.push({x, y: fTrue(x) + noise});
        }

        this.computeAllDegreesErrors();
    }

    fitPolynomial(degree, lambdaVal) {
        const d = degree;
        const n = this.trainPoints.length;
        const m = d + 1;

        const ATA = Array.from({length: m}, () => new Float64Array(m));
        const ATy = new Float64Array(m);

        for (let i = 0; i < n; i++) {
            const {x, y} = this.trainPoints[i];
            const pows = new Float64Array(2 * d + 1);
            pows[0] = 1;
            for (let p = 1; p <= 2 * d; p++) {
                pows[p] = pows[p - 1] * x;
            }

            for (let r = 0; r < m; r++) {
                for (let c = 0; c < m; c++) {
                    ATA[r][c] += pows[r + c];
                }
                ATy[r] += pows[r] * y;
            }
        }

        for (let r = 1; r < m; r++) {
            ATA[r][r] += lambdaVal;
        }

        for (let i = 0; i < m; i++) {
            let maxRow = i;
            for (let k = i + 1; k < m; k++) {
                if (Math.abs(ATA[k][i]) > Math.abs(ATA[maxRow][i])) {
                    maxRow = k;
                }
            }
            const tmpRow = ATA[i];
            ATA[i] = ATA[maxRow];
            ATA[maxRow] = tmpRow;
            const tmpY = ATy[i];
            ATy[i] = ATy[maxRow];
            ATy[maxRow] = tmpY;

            if (Math.abs(ATA[i][i]) < 1e-12) {
                continue;
            }

            for (let k = i + 1; k < m; k++) {
                const factor = ATA[k][i] / ATA[i][i];
                ATy[k] -= factor * ATy[i];
                for (let j = i; j < m; j++) {
                    ATA[k][j] -= factor * ATA[i][j];
                }
            }
        }

        const w = new Float64Array(m);
        for (let i = m - 1; i >= 0; i--) {
            let sum = ATy[i];
            for (let j = i + 1; j < m; j++) {
                sum -= ATA[i][j] * w[j];
            }
            w[i] = ATA[i][i] !== 0 ? sum / ATA[i][i] : 0;
        }

        return w;
    }

    evalPoly(w, x) {
        let val = 0;
        let px = 1;
        for (let i = 0; i < w.length; i++) {
            val += w[i] * px;
            px *= x;
        }
        return val;
    }

    calcMSE(w, pts) {
        let sum = 0;
        for (const pt of pts) {
            const pred = this.evalPoly(w, pt.x);
            sum += Math.pow(pred - pt.y, 2);
        }
        return sum / pts.length;
    }

    computeAllDegreesErrors() {
        this.degreeErrors = [];
        for (let d = 1; d <= 9; d++) {
            const w = this.fitPolynomial(d, this.lambda);
            const trainErr = this.calcMSE(w, this.trainPoints);
            const testErr = this.calcMSE(w, this.testPoints);
            this.degreeErrors.push({d, trainErr, testErr});
        }
    }

    renderLayout() {
        const isSine = this.datasetKey === 'sine' ? 'selected' : '';
        const isQuad = this.datasetKey === 'quadratic' ? 'selected' : '';
        const isEng = this.datasetKey === 'engine' ? 'selected' : '';

        this.container.innerHTML = `
            <div class="aisim-container aisim-regression">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <div class="aisim-dataset-select">
                        <select class="aisim-select" id="aisim-reg-dataset">
                            <option value="sine" ${isSine}>${this.t.datasets.sine}</option>
                            <option value="quadratic" ${isQuad}>${this.t.datasets.quadratic}</option>
                            <option value="engine" ${isEng}>${this.t.datasets.engine}</option>
                        </select>
                    </div>
                </div>

                <div class="aisim-body">
                    <div class="aisim-slider-card">
                        <div class="aisim-th-header">
                            <label for="aisim-reg-degree">
                                <strong>${this.t.degreeLabel}</strong>
                                <span class="aisim-th-pill" id="aisim-reg-degree-val">d = ${this.degree}</span>
                            </label>
                            <span class="aisim-th-sub">
                                ${this.tr('1 = Prava (Underfit) ... 9 = Polinom (Overfit)', '1 = Linear ... 9 = Overfitting')}
                            </span>
                        </div>
                        <input type="range" class="aisim-slider aisim-slider-large" id="aisim-reg-degree"
                               min="1" max="9" step="1" value="${this.degree}">
                    </div>

                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box">
                                <div class="aisim-canvas-header">
                                    <span>Prilagođavanje modela ŷ(x)</span>
                                    <div class="aisim-legend">
                                        <span class="aisim-legend-item">
                                            <span class="aisim-dot aisim-dot-neg"></span> ${this.t.trainPts}
                                        </span>
                                        <span class="aisim-legend-item">
                                            <span class="aisim-dot aisim-dot-pos"></span> ${this.t.testPts}
                                        </span>
                                        <span class="aisim-legend-item">
                                            <span class="aisim-line-th" style="background:#2563eb; height:3px;"></span>
                                            ${this.t.polyCurve}
                                        </span>
                                    </div>
                                </div>
                                <canvas class="aisim-canvas" id="aisim-reg-canvas"></canvas>
                            </div>

                            <div class="aisim-card aisim-advice-card" id="aisim-reg-advice">
                                <div class="aisim-advice-icon" id="aisim-reg-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-reg-advice-text">...</div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.regLabel}</div>
                                <div class="aisim-slider-group">
                                    <label>λ = <span id="aisim-reg-lambda-val">${this.lambda.toFixed(3)}</span></label>
                                    <input type="range" class="aisim-slider" id="aisim-reg-lambda"
                                           min="0.0" max="0.5" step="0.01" value="${this.lambda}">
                                </div>
                                <div style="font-size:0.78rem; color:#64748b; line-height:1.4;">
                                    ${this.tr(
                                        'L2 (Ridge) regularizacija kažnjava velike težine i smanjuje overfit.',
                                        'L2 (Ridge) penalizes large weights, smoothing out overfitting.'
                                    )}
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">Greška (MSE)</div>
                                <div class="aisim-stat-row">
                                    <span style="color:#2563eb; font-weight:600;">Train MSE:</span>
                                    <span class="aisim-stat-val" id="val-train-mse">0.000</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span style="color:#ef4444; font-weight:600;">Validation MSE:</span>
                                    <span class="aisim-stat-val" id="val-test-mse">0.000</span>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.complexityPlotTitle}</div>
                                <canvas class="aisim-mini-chart" id="aisim-reg-u-canvas"></canvas>
                                <div class="aisim-u-legend"
                                     style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b;">
                                    <span>Stepen: 1</span>
                                    <span style="color:#2563eb;">■ Train</span>
                                    <span style="color:#ef4444;">■ Test</span>
                                    <span>9</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.canvas = this.container.querySelector('#aisim-reg-canvas');
        this.uCanvas = this.container.querySelector('#aisim-reg-u-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.uCtx = this.uCanvas.getContext('2d');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;

        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = (rect.width || 540) * dpr;
        this.canvas.height = (rect.height || 280) * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvasWidth = rect.width || 540;
        this.canvasHeight = rect.height || 280;

        const uRect = this.uCanvas.getBoundingClientRect();
        this.uCanvas.width = (uRect.width || 280) * dpr;
        this.uCanvas.height = (uRect.height || 130) * dpr;
        this.uCtx.scale(dpr, dpr);
        this.uWidth = uRect.width || 280;
        this.uHeight = uRect.height || 130;
    }

    bindEvents() {
        const sliderDegree = this.container.querySelector('#aisim-reg-degree');
        const sliderLambda = this.container.querySelector('#aisim-reg-lambda');
        const selDataset = this.container.querySelector('#aisim-reg-dataset');

        if (this.readonly) {
            if (sliderDegree) {
                sliderDegree.disabled = true;
            }
            if (sliderLambda) {
                sliderLambda.disabled = true;
            }
            if (selDataset) {
                selDataset.disabled = true;
            }
            return;
        }

        sliderDegree.addEventListener('input', (e) => {
            this.degree = parseInt(e.target.value, 10);
            this.container.querySelector('#aisim-reg-degree-val').textContent = `d = ${this.degree}`;
            this.update();
        });

        sliderLambda.addEventListener('input', (e) => {
            this.lambda = parseFloat(e.target.value);
            this.container.querySelector('#aisim-reg-lambda-val').textContent = this.lambda.toFixed(3);
            this.computeAllDegreesErrors();
            this.update();
        });

        selDataset.addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.initData();
            this.renderLayout();
            this.bindEvents();
            this.update();
        });

        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const padX = 40;
            const padY = 30;
            const gw = this.canvasWidth - 2 * padX;
            const gh = this.canvasHeight - 2 * padY;

            const x = -2.0 + ((clickX - padX) / gw) * 4.0;
            const y = 2.5 - ((clickY - padY) / gh) * 5.0;

            if (x >= -2.0 && x <= 2.0 && y >= -2.5 && y <= 2.5) {
                this.trainPoints.push({x, y});
                this.computeAllDegreesErrors();
                this.update();
            }
        });

        window.addEventListener('resize', () => {
            this.setupDPI();
            this.update();
        });
    }

    update() {
        this.weights = this.fitPolynomial(this.degree, this.lambda);
        this.trainMSE = this.calcMSE(this.weights, this.trainPoints);
        this.testMSE = this.calcMSE(this.weights, this.testPoints);

        this.container.querySelector('#val-train-mse').textContent = this.trainMSE.toFixed(4);
        this.container.querySelector('#val-test-mse').textContent = this.testMSE.toFixed(4);

        this.updateAdvice();
        this.drawRegressionPlot();
        this.drawUCurve();

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'regression',
                parameters: {
                    degree: this.degree,
                    lambda: this.lambda,
                    dataset: this.datasetKey
                },
                metrics: {
                    trainMse: Math.round(this.trainMSE * 10000) / 10000,
                    valMse: Math.round(this.testMSE * 10000) / 10000
                }
            });
        }
    }

    updateAdvice() {
        const adviceBox = this.container.querySelector('#aisim-reg-advice-text');
        const icon = this.container.querySelector('#aisim-reg-icon');

        if (this.degree === 1) {
            icon.textContent = '⚠️';
            adviceBox.textContent = this.t.underfitting;
        } else if (this.degree >= 6 && this.lambda < 0.03 && this.testMSE > this.trainMSE * 2.5) {
            icon.textContent = '🚨';
            adviceBox.textContent = this.t.overfitting;
        } else {
            icon.textContent = '✅';
            adviceBox.textContent = this.t.optimal;
        }
    }

    toPx(x, y) {
        const padX = 40;
        const padY = 30;
        const gw = this.canvasWidth - 2 * padX;
        const gh = this.canvasHeight - 2 * padY;
        const px = padX + ((x - (-2.0)) / 4.0) * gw;
        const py = (this.canvasHeight - padY) - ((y - (-2.5)) / 5.0) * gh;
        return {px, py};
    }

    drawRegressionPlot() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;
        ctx.clearRect(0, 0, w, h);

        const padX = 40;
        const padY = 30;

        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        const origin = this.toPx(0, 0);

        ctx.beginPath();
        ctx.moveTo(padX, origin.py);
        ctx.lineTo(w - padX, origin.py);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(origin.px, padY);
        ctx.lineTo(origin.px, h - padY);
        ctx.stroke();

        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const steps = 150;
        for (let i = 0; i <= steps; i++) {
            const x = -2.0 + (4.0 / steps) * i;
            const y = this.evalPoly(this.weights, x);
            const {px, py} = this.toPx(x, Math.max(-3.5, Math.min(3.5, y)));
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        for (const pt of this.trainPoints) {
            const {px, py} = this.toPx(pt.x, pt.y);
            ctx.fillStyle = '#2563eb';
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        for (const pt of this.testPoints) {
            const {px, py} = this.toPx(pt.x, pt.y);
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.rect(px - 3.5, py - 3.5, 7, 7);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.stroke();
        }
    }

    drawUCurve() {
        const ctx = this.uCtx;
        const w = this.uWidth;
        const h = this.uHeight;
        ctx.clearRect(0, 0, w, h);

        const pad = 20;
        const gw = w - 2 * pad;
        const gh = h - 2 * pad;

        const allErr = this.degreeErrors.map(e => Math.min(4.0, Math.max(e.trainErr, e.testErr)));
        const maxErr = Math.min(4.0, Math.max(...allErr));

        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.degreeErrors.length; i++) {
            const item = this.degreeErrors[i];
            const px = pad + (i / 8) * gw;
            const normY = Math.min(item.trainErr, maxErr) / maxErr;
            const py = (h - pad) - normY * gh;
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.degreeErrors.length; i++) {
            const item = this.degreeErrors[i];
            const px = pad + (i / 8) * gw;
            const normY = Math.min(item.testErr, maxErr) / maxErr;
            const py = (h - pad) - normY * gh;
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        const curPx = pad + ((this.degree - 1) / 8) * gw;
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(curPx, pad);
        ctx.lineTo(curPx, h - pad);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}
