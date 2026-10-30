/**
 * Confusion Matrix & Classification Threshold interactive simulation engine.
 *
 * @module filter_aisim/sims/confusion_matrix
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

export class ConfusionMatrixSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('cm_title'),
            badge: this.str('cm_badge'),
            scenario: this.str('cm_scenario'),
            thresholdLabel: this.str('cm_threshold_label'),
            probDistTitle: this.str('cm_prob_dist_title'),
            matrixTitle: this.str('cm_matrix_title'),
            metricsTitle: this.str('cm_metrics_title'),
            rocTitle: this.str('cm_roc_title'),
            actualPos: this.str('cm_actual_pos'),
            actualNeg: this.str('cm_actual_neg'),
            predPos: this.str('cm_pred_pos'),
            predNeg: this.str('cm_pred_neg'),
            tp: this.str('cm_tp'),
            fn: this.str('cm_fn'),
            fp: this.str('cm_fp'),
            tn: this.str('cm_tn'),
            accuracy: this.str('cm_accuracy'),
            precision: this.str('cm_precision'),
            recall: this.str('cm_recall'),
            f1: this.str('cm_f1'),
            scenarios: {
                quality: {
                    name: this.str('cm_scenarios_quality_name'),
                    posLabel: this.str('cm_scenarios_quality_pos_label'),
                    negLabel: this.str('cm_scenarios_quality_neg_label'),
                    hintLow: this.str('cm_scenarios_quality_hint_low'),
                    hintHigh: this.str('cm_scenarios_quality_hint_high')
                },
                medical: {
                    name: this.str('cm_scenarios_medical_name'),
                    posLabel: this.str('cm_scenarios_medical_pos_label'),
                    negLabel: this.str('cm_scenarios_medical_neg_label'),
                    hintLow: this.str('cm_scenarios_medical_hint_low'),
                    hintHigh: this.str('cm_scenarios_medical_hint_high')
                },
                spam: {
                    name: this.str('cm_scenarios_spam_name'),
                    posLabel: this.str('cm_scenarios_spam_pos_label'),
                    negLabel: this.str('cm_scenarios_spam_neg_label'),
                    hintLow: this.str('cm_scenarios_spam_hint_low'),
                    hintHigh: this.str('cm_scenarios_spam_hint_high')
                }
            }
        };

        this.scenarioKey = options.scenario || 'quality';
        this.threshold = parseFloat(options.threshold) || 0.50;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        // Generate synthetic distributions for dataset.
        this.initDataset();

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

    initDataset() {
        this.samples = [];
        const nNeg = 120;
        const nPos = 40;

        for (let i = 0; i < nNeg; i++) {
            const p = Math.max(0.01, Math.min(0.99, this.randomGaussian(0.30, 0.14)));
            this.samples.push({actual: 0, prob: p});
        }

        for (let i = 0; i < nPos; i++) {
            const p = Math.max(0.01, Math.min(0.99, this.randomGaussian(0.70, 0.15)));
            this.samples.push({actual: 1, prob: p});
        }

        this.computeROCCurve();
    }

    randomGaussian(mean, std) {
        let u = 0;
        let v = 0;
        while (u === 0) {
            u = Math.random();
        }
        while (v === 0) {
            v = Math.random();
        }
        const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return num * std + mean;
    }

    computeROCCurve() {
        this.rocPoints = [];
        const steps = 100;
        for (let i = 0; i <= steps; i++) {
            const th = i / steps;
            const m = this.calcMatrixAt(th);
            const fpr = (m.fp + m.tn) > 0 ? (m.fp / (m.fp + m.tn)) : 0;
            const tpr = (m.tp + m.fn) > 0 ? (m.tp / (m.tp + m.fn)) : 0;
            this.rocPoints.push({th, fpr, tpr});
        }

        const sorted = [...this.rocPoints].sort((a, b) => a.fpr - b.fpr);
        let auc = 0;
        for (let i = 1; i < sorted.length; i++) {
            const dx = sorted[i].fpr - sorted[i - 1].fpr;
            const avgY = (sorted[i].tpr + sorted[i - 1].tpr) / 2;
            auc += dx * avgY;
        }
        this.auc = Math.max(0.5, Math.min(1.0, auc));
    }

    calcMatrixAt(th) {
        let tp = 0;
        let fp = 0;
        let tn = 0;
        let fn = 0;
        for (const s of this.samples) {
            const pred = s.prob >= th ? 1 : 0;
            if (s.actual === 1 && pred === 1) {
                tp++;
            } else if (s.actual === 0 && pred === 1) {
                fp++;
            } else if (s.actual === 0 && pred === 0) {
                tn++;
            } else if (s.actual === 1 && pred === 0) {
                fn++;
            }
        }
        return {tp, fp, tn, fn, total: this.samples.length};
    }

    renderLayout() {
        const sc = this.t.scenarios[this.scenarioKey];
        const isQ = this.scenarioKey === 'quality' ? 'selected' : '';
        const isM = this.scenarioKey === 'medical' ? 'selected' : '';
        const isS = this.scenarioKey === 'spam' ? 'selected' : '';

        this.container.innerHTML = `
            <div class="aisim-container aisim-confmat">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <div class="aisim-scenario-select">
                        <select class="aisim-select" id="aisim-cm-scenario">
                            <option value="quality" ${isQ}>${this.t.scenarios.quality.name}</option>
                            <option value="medical" ${isM}>${this.t.scenarios.medical.name}</option>
                            <option value="spam" ${isS}>${this.t.scenarios.spam.name}</option>
                        </select>
                    </div>
                </div>

                <div class="aisim-body">
                    <div class="aisim-slider-card">
                        <div class="aisim-th-header">
                            <label for="aisim-cm-th">
                                <strong>${this.t.thresholdLabel}</strong>
                                <span class="aisim-th-pill" id="aisim-cm-th-val">${this.threshold.toFixed(2)}</span>
                            </label>
                            <span class="aisim-th-sub">
                                ${this.str('cm_drag_hint')}
                            </span>
                        </div>
                        <input type="range" class="aisim-slider aisim-slider-large" id="aisim-cm-th"
                               min="0.05" max="0.95" step="0.01" value="${this.threshold}">
                    </div>

                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box">
                                <div class="aisim-canvas-header">
                                    <span>${this.t.probDistTitle}</span>
                                    <div class="aisim-legend">
                                        <span class="aisim-legend-item">
                                            <span class="aisim-dot aisim-dot-neg"></span> ${sc.negLabel} (0)
                                        </span>
                                        <span class="aisim-legend-item">
                                            <span class="aisim-dot aisim-dot-pos"></span> ${sc.posLabel} (1)
                                        </span>
                                        <span class="aisim-legend-item">
                                            <span class="aisim-line-th"></span> ${this.str('cm_threshold_short')}
                                        </span>
                                    </div>
                                </div>
                                <canvas class="aisim-canvas" id="aisim-cm-dist-canvas"></canvas>
                            </div>

                            <div class="aisim-card aisim-advice-card" id="aisim-cm-advice">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-cm-advice-text">...</div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.matrixTitle}</div>
                                <div class="aisim-matrix-grid">
                                    <div class="aisim-matrix-corner"></div>
                                    <div class="aisim-matrix-head">${this.t.predPos}</div>
                                    <div class="aisim-matrix-head">${this.t.predNeg}</div>

                                    <div class="aisim-matrix-rowhead">${this.t.actualPos}</div>
                                    <div class="aisim-cell aisim-cell-tp" id="aisim-cell-tp">
                                        <div class="aisim-cell-lbl">${this.t.tp}</div>
                                        <div class="aisim-cell-val" id="val-tp">0</div>
                                    </div>
                                    <div class="aisim-cell aisim-cell-fn" id="aisim-cell-fn">
                                        <div class="aisim-cell-lbl">${this.t.fn}</div>
                                        <div class="aisim-cell-val" id="val-fn">0</div>
                                    </div>

                                    <div class="aisim-matrix-rowhead">${this.t.actualNeg}</div>
                                    <div class="aisim-cell aisim-cell-fp" id="aisim-cell-fp">
                                        <div class="aisim-cell-lbl">${this.t.fp}</div>
                                        <div class="aisim-cell-val" id="val-fp">0</div>
                                    </div>
                                    <div class="aisim-cell aisim-cell-tn" id="aisim-cell-tn">
                                        <div class="aisim-cell-lbl">${this.t.tn}</div>
                                        <div class="aisim-cell-val" id="val-tn">0</div>
                                    </div>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.metricsTitle}</div>
                                <div class="aisim-metric-row">
                                    <div class="aisim-metric-header">
                                        <span>${this.t.accuracy}</span>
                                        <strong id="val-acc">0%</strong>
                                    </div>
                                    <div class="aisim-prog-bar"><div class="aisim-prog-fill" id="bar-acc"></div></div>
                                </div>
                                <div class="aisim-metric-row">
                                    <div class="aisim-metric-header">
                                        <span>${this.t.precision}</span>
                                        <strong id="val-prec">0%</strong>
                                    </div>
                                    <div class="aisim-prog-bar">
                                        <div class="aisim-prog-fill aisim-fill-prec" id="bar-prec"></div>
                                    </div>
                                </div>
                                <div class="aisim-metric-row">
                                    <div class="aisim-metric-header">
                                        <span>${this.t.recall}</span>
                                        <strong id="val-rec">0%</strong>
                                    </div>
                                    <div class="aisim-prog-bar">
                                        <div class="aisim-prog-fill aisim-fill-rec" id="bar-rec"></div>
                                    </div>
                                </div>
                                <div class="aisim-metric-row">
                                    <div class="aisim-metric-header">
                                        <span>${this.t.f1}</span>
                                        <strong id="val-f1">0%</strong>
                                    </div>
                                    <div class="aisim-prog-bar">
                                        <div class="aisim-prog-fill aisim-fill-f1" id="bar-f1"></div>
                                    </div>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">
                                    <span>${this.t.rocTitle}</span>
                                    <span class="aisim-auc-badge">AUC = ${this.auc.toFixed(3)}</span>
                                </div>
                                <canvas class="aisim-mini-chart" id="aisim-cm-roc-canvas"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.distCanvas = this.container.querySelector('#aisim-cm-dist-canvas');
        this.rocCanvas = this.container.querySelector('#aisim-cm-roc-canvas');
        this.distCtx = this.distCanvas.getContext('2d');
        this.rocCtx = this.rocCanvas.getContext('2d');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;

        const rectDist = this.distCanvas.getBoundingClientRect();
        this.distCanvas.width = (rectDist.width || 540) * dpr;
        this.distCanvas.height = (rectDist.height || 280) * dpr;
        this.distCtx.scale(dpr, dpr);
        this.distWidth = rectDist.width || 540;
        this.distHeight = rectDist.height || 280;

        const rectRoc = this.rocCanvas.getBoundingClientRect();
        this.rocCanvas.width = (rectRoc.width || 280) * dpr;
        this.rocCanvas.height = (rectRoc.height || 140) * dpr;
        this.rocCtx.scale(dpr, dpr);
        this.rocWidth = rectRoc.width || 280;
        this.rocHeight = rectRoc.height || 140;
    }

    bindEvents() {
        const sliderTh = this.container.querySelector('#aisim-cm-th');
        const selScenario = this.container.querySelector('#aisim-cm-scenario');

        if (this.readonly) {
            if (sliderTh) {
                sliderTh.disabled = true;
            }
            if (selScenario) {
                selScenario.disabled = true;
            }
            return;
        }

        sliderTh.addEventListener('input', (e) => {
            this.threshold = parseFloat(e.target.value);
            this.container.querySelector('#aisim-cm-th-val').textContent = this.threshold.toFixed(2);
            this.update();
        });

        selScenario.addEventListener('change', (e) => {
            this.scenarioKey = e.target.value;
            this.initDataset();
            this.renderLayout();
            this.bindEvents();
            this.update();
        });

        const setThFromEvent = (e) => {
            const rect = this.distCanvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const padX = 40;
            const usableW = this.distWidth - 2 * padX;
            const ratio = (clickX - padX) / usableW;
            const newTh = Math.max(0.05, Math.min(0.95, ratio));
            this.threshold = newTh;
            sliderTh.value = newTh;
            this.container.querySelector('#aisim-cm-th-val').textContent = newTh.toFixed(2);
            this.update();
        };

        let isDragging = false;
        this.distCanvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            setThFromEvent(e);
        });
        window.addEventListener('mousemove', (e) => {
            if (isDragging) {
                setThFromEvent(e);
            }
        });
        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        window.addEventListener('resize', () => {
            this.setupDPI();
            this.update();
        });
    }

    update() {
        const m = this.calcMatrixAt(this.threshold);
        this.updateMatrixUI(m);
        this.updateMetricsUI(m);
        this.updateAdviceUI(m);
        this.drawDistribution();
        this.drawROC(m);

        if (typeof this.onChange === 'function') {
            const acc = (m.tp + m.tn) / (m.total || 1);
            const prec = (m.tp + m.fp) > 0 ? (m.tp / (m.tp + m.fp)) : 0;
            const rec = (m.tp + m.fn) > 0 ? (m.tp / (m.tp + m.fn)) : 0;
            const f1 = (prec + rec) > 0 ? (2 * prec * rec / (prec + rec)) : 0;
            this.onChange({
                simtype: 'confusion_matrix',
                parameters: {
                    scenario: this.scenarioKey,
                    threshold: this.threshold
                },
                metrics: {
                    tp: m.tp,
                    fp: m.fp,
                    tn: m.tn,
                    fn: m.fn,
                    accuracy: Math.round(acc * 1000) / 1000,
                    precision: Math.round(prec * 1000) / 1000,
                    recall: Math.round(rec * 1000) / 1000,
                    f1: Math.round(f1 * 1000) / 1000
                }
            });
        }
    }

    updateMatrixUI(m) {
        this.container.querySelector('#val-tp').textContent = m.tp;
        this.container.querySelector('#val-fn').textContent = m.fn;
        this.container.querySelector('#val-fp').textContent = m.fp;
        this.container.querySelector('#val-tn').textContent = m.tn;
    }

    updateMetricsUI(m) {
        const acc = (m.tp + m.tn) / m.total;
        const prec = (m.tp + m.fp) > 0 ? (m.tp / (m.tp + m.fp)) : 0;
        const rec = (m.tp + m.fn) > 0 ? (m.tp / (m.tp + m.fn)) : 0;
        const f1 = (prec + rec) > 0 ? (2 * prec * rec / (prec + rec)) : 0;

        const setMetric = (idVal, idBar, val) => {
            const pct = Math.round(val * 100);
            this.container.querySelector(idVal).textContent = `${pct}%`;
            this.container.querySelector(idBar).style.width = `${pct}%`;
        };

        setMetric('#val-acc', '#bar-acc', acc);
        setMetric('#val-prec', '#bar-prec', prec);
        setMetric('#val-rec', '#bar-rec', rec);
        setMetric('#val-f1', '#bar-f1', f1);
    }

    updateAdviceUI(m) {
        const sc = this.t.scenarios[this.scenarioKey];
        const adviceBox = this.container.querySelector('#aisim-cm-advice-text');
        if (this.threshold < 0.35) {
            adviceBox.textContent = sc.hintLow;
        } else if (this.threshold > 0.65) {
            adviceBox.textContent = sc.hintHigh;
        } else {
            const precVal = Math.round((m.tp / (m.tp + m.fp || 1)) * 100);
            const recVal = Math.round((m.tp / (m.tp + m.fn || 1)) * 100);
            adviceBox.textContent = this.str('cm_hint_balanced', {precval: precVal, recval: recVal});
        }
    }

    drawDistribution() {
        const ctx = this.distCtx;
        const w = this.distWidth;
        const h = this.distHeight;
        ctx.clearRect(0, 0, w, h);

        const padX = 40;
        const padY = 30;
        const graphW = w - 2 * padX;
        const graphH = h - 2 * padY;

        // Draw axes.
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padX, h - padY);
        ctx.lineTo(w - padX, h - padY);
        ctx.stroke();

        // X-axis ticks (0.0 to 1.0).
        ctx.fillStyle = '#64748b';
        ctx.font = '11px sans-serif';
        for (let i = 0; i <= 10; i += 2) {
            const val = i / 10;
            const px = padX + val * graphW;
            ctx.beginPath();
            ctx.moveTo(px, h - padY);
            ctx.lineTo(px, h - padY + 4);
            ctx.stroke();
            ctx.fillText(val.toFixed(1), px - 9, h - padY + 16);
        }

        // Draw individual data dots at bottom.
        for (const s of this.samples) {
            const px = padX + s.prob * graphW;
            const py = (s.actual === 0 ? h - padY - 8 : h - padY - 18) - (Math.random() * 8);
            ctx.fillStyle = s.actual === 1 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(37, 99, 235, 0.6)';
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Density function approximation.
        const gaussian = (x, mean, std) => {
            return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
        };

        const drawBell = (mean, std, strokeColor, fillColor) => {
            ctx.beginPath();
            const steps = 80;
            for (let i = 0; i <= steps; i++) {
                const x = i / steps;
                const density = gaussian(x, mean, std);
                const px = padX + x * graphW;
                const py = (h - padY - 30) - (density / 3.0) * (graphH - 40);
                if (i === 0) {
                    ctx.moveTo(px, h - padY - 30);
                } else {
                    ctx.lineTo(px, py);
                }
            }
            ctx.lineTo(padX + graphW, h - padY - 30);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.stroke();
        };

        // Draw negative and positive class distributions.
        drawBell(0.30, 0.14, '#2563eb', 'rgba(37, 99, 235, 0.15)');
        drawBell(0.70, 0.15, '#ef4444', 'rgba(239, 68, 68, 0.15)');

        // Draw vertical threshold line θ.
        const thPx = padX + this.threshold * graphW;
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(thPx, 15);
        ctx.lineTo(thPx, h - padY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Handle on top of threshold line.
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(thPx, 18, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(thPx, 18, 2.5, 0, Math.PI * 2);
        ctx.fill();
    }

    drawROC(currentMatrix) {
        const ctx = this.rocCtx;
        const w = this.rocWidth;
        const h = this.rocHeight;
        ctx.clearRect(0, 0, w, h);

        const pad = 24;
        const gw = w - 2 * pad;
        const gh = h - 2 * pad;

        // Diagonal baseline (random guess).
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad, h - pad);
        ctx.lineTo(pad + gw, pad);
        ctx.stroke();

        // ROC Curve line.
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const sorted = [...this.rocPoints].sort((a, b) => a.fpr - b.fpr);
        for (let i = 0; i < sorted.length; i++) {
            const pt = sorted[i];
            const px = pad + pt.fpr * gw;
            const py = (h - pad) - pt.tpr * gh;
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Current point on ROC.
        const m = currentMatrix;
        const curFpr = (m.fp + m.tn) > 0 ? (m.fp / (m.fp + m.tn)) : 0;
        const curTpr = (m.tp + m.fn) > 0 ? (m.tp / (m.tp + m.fn)) : 0;
        const curPx = pad + curFpr * gw;
        const curPy = (h - pad) - curTpr * gh;

        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(curPx, curPy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Axis labels.
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px sans-serif';
        ctx.fillText('FPR (1 - Spec)', pad + gw - 65, h - pad + 14);
        ctx.fillText('TPR (Recall)', pad, pad - 6);
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
ConfusionMatrixSim.STRINGS = [
    'cm_title', 'cm_badge', 'cm_scenario', 'cm_threshold_label', 'cm_prob_dist_title', 'cm_matrix_title',
    'cm_metrics_title', 'cm_roc_title', 'cm_actual_pos', 'cm_actual_neg', 'cm_pred_pos', 'cm_pred_neg', 'cm_tp',
    'cm_fn', 'cm_fp', 'cm_tn', 'cm_accuracy', 'cm_precision', 'cm_recall', 'cm_f1', 'cm_scenarios_quality_name',
    'cm_scenarios_quality_pos_label', 'cm_scenarios_quality_neg_label', 'cm_scenarios_quality_hint_low',
    'cm_scenarios_quality_hint_high', 'cm_scenarios_medical_name', 'cm_scenarios_medical_pos_label',
    'cm_scenarios_medical_neg_label', 'cm_scenarios_medical_hint_low', 'cm_scenarios_medical_hint_high',
    'cm_scenarios_spam_name', 'cm_scenarios_spam_pos_label', 'cm_scenarios_spam_neg_label',
    'cm_scenarios_spam_hint_low', 'cm_scenarios_spam_hint_high', 'cm_drag_hint', 'cm_threshold_short',
    'cm_hint_balanced'
];
