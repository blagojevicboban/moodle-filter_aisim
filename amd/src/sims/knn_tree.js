/**
 * Decision Boundaries 2D: k-NN and Decision Tree interactive simulation engine.
 *
 * @module filter_aisim/sims/knn_tree
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

export class KnnTreeSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('knn_title'),
            badge: this.str('knn_badge'),
            modelLabel: this.str('knn_model_label'),
            knnModel: this.str('knn_knn_model'),
            treeModel: this.str('knn_tree_model'),
            datasetLabel: this.str('knn_dataset_label'),
            kParam: this.str('knn_k_param'),
            depthParam: this.str('knn_depth_param'),
            classAPts: this.str('knn_class_apts'),
            classBPts: this.str('knn_class_bpts'),
            addPointHint: this.str('knn_add_point_hint'),
            activeClassLabel: this.str('knn_active_class_label'),
            accuracy: this.str('knn_accuracy'),
            boundaryNotice: this.str('knn_boundary_notice'),
            knnNotice: this.str('knn_knn_notice'),
            treeNotice: this.str('knn_tree_notice'),
            datasets: {
                moons: this.str('knn_datasets_moons'),
                circles: this.str('knn_datasets_circles'),
                blobs: this.str('knn_datasets_blobs')
            }
        };

        this.modelType = options.model || 'knn'; // 'knn' or 'tree'
        this.datasetKey = options.dataset || 'moons';
        this.k = parseInt(options.k, 10) || 3;
        this.maxDepth = parseInt(options.depth, 10) || 3;
        this.activeClass = 0; // 0 = Blue (Class A), 1 = Red (Class B)
        this.hoverPoint = null;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;

        this.initData();
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

    initData() {
        this.points = [];
        let seed = 123;
        const rand = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return (seed / 233280) * 2 - 1;
        };

        if (this.datasetKey === 'moons') {
            const n = 24;
            for (let i = 0; i < n; i++) {
                const angle = (Math.PI / (n - 1)) * i;
                const x1 = Math.cos(angle) * 1.1 + (rand() * 0.12);
                const y1 = Math.sin(angle) * 1.1 + (rand() * 0.12) - 0.2;
                this.points.push({x: x1, y: y1, cls: 0});

                const x2 = 1.0 - Math.cos(angle) * 1.1 + (rand() * 0.12) - 0.5;
                const y2 = 0.5 - Math.sin(angle) * 1.1 + (rand() * 0.12);
                this.points.push({x: x2, y: y2, cls: 1});
            }
        } else if (this.datasetKey === 'circles') {
            const nOuter = 26;
            const nInner = 16;
            for (let i = 0; i < nOuter; i++) {
                const angle = (2 * Math.PI / nOuter) * i;
                const r = 1.3 + (rand() * 0.12);
                this.points.push({x: Math.cos(angle) * r, y: Math.sin(angle) * r, cls: 0});
            }
            for (let i = 0; i < nInner; i++) {
                const angle = (2 * Math.PI / nInner) * i;
                const r = 0.55 + (rand() * 0.1);
                this.points.push({x: Math.cos(angle) * r, y: Math.sin(angle) * r, cls: 1});
            }
        } else {
            // Blobs
            for (let i = 0; i < 20; i++) {
                this.points.push({x: -0.7 + rand() * 0.55, y: 0.5 + rand() * 0.55, cls: 0});
                this.points.push({x: 0.7 + rand() * 0.55, y: -0.5 + rand() * 0.55, cls: 1});
            }
        }
    }

    renderLayout() {
        const isKnn = this.modelType === 'knn';
        const isM = this.datasetKey === 'moons' ? 'selected' : '';
        const isC = this.datasetKey === 'circles' ? 'selected' : '';
        const isB = this.datasetKey === 'blobs' ? 'selected' : '';

        this.container.innerHTML = `
            <div class="aisim-container aisim-knntree">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <div style="display:flex; gap:8px;">
                        <select class="aisim-select" id="aisim-kt-dataset">
                            <option value="moons" ${isM}>${this.t.datasets.moons}</option>
                            <option value="circles" ${isC}>${this.t.datasets.circles}</option>
                            <option value="blobs" ${isB}>${this.t.datasets.blobs}</option>
                        </select>
                    </div>
                </div>

                <div class="aisim-body">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div class="sim-tabs">
                            <button class="tab-btn ${isKnn ? 'active' : ''}" id="btn-tab-knn">
                                🔵 ${this.t.knnModel}
                            </button>
                            <button class="tab-btn ${!isKnn ? 'active' : ''}" id="btn-tab-tree">
                                🌲 ${this.t.treeModel}
                            </button>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px; font-size:0.85rem;">
                            <span style="font-weight:600;">${this.t.activeClassLabel}</span>
                            <button class="aisim-btn ${this.activeClass === 0 ? 'aisim-btn-primary' : 'aisim-btn-secondary'}"
                                    id="btn-cls-a" style="padding:3px 10px;">
                                ● ${this.t.classAPts}
                            </button>
                            <button class="aisim-btn ${this.activeClass === 1 ? 'aisim-btn-danger' : 'aisim-btn-secondary'}"
                                    id="btn-cls-b" style="padding:3px 10px;">
                                ● ${this.t.classBPts}
                            </button>
                        </div>
                    </div>

                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box" style="padding:0; overflow:hidden;">
                                <canvas class="aisim-canvas" id="aisim-kt-canvas"
                                        style="height:320px; cursor:crosshair;"></canvas>
                            </div>

                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-kt-advice">
                                    ${isKnn ? this.t.knnNotice : this.t.treeNotice}
                                </div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card" id="card-param-knn" style="display:${isKnn ? 'block' : 'none'};">
                                <div class="aisim-card-title">${this.t.kParam}</div>
                                <div class="aisim-slider-group">
                                    <label>k = <span id="val-k">${this.k}</span></label>
                                    <input type="range" class="aisim-slider" id="slider-k"
                                           min="1" max="15" step="2" value="${this.k}">
                                </div>
                                <div style="font-size:0.78rem; color:#64748b; line-height:1.4;">
                                    ${this.str('knn_k_hint')}
                                </div>
                            </div>

                            <div class="aisim-card" id="card-param-tree" style="display:${!isKnn ? 'block' : 'none'};">
                                <div class="aisim-card-title">${this.t.depthParam}</div>
                                <div class="aisim-slider-group">
                                    <label>${this.str('knn_depth')} = <span id="val-depth">${this.maxDepth}</span></label>
                                    <input type="range" class="aisim-slider" id="slider-depth"
                                           min="1" max="5" step="1" value="${this.maxDepth}">
                                </div>
                                <div style="font-size:0.78rem; color:#64748b; line-height:1.4;">
                                    ${this.str('knn_depth_hint')}
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.str('knn_metrics')}</div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.accuracy}</span>
                                    <span class="aisim-stat-val" id="val-kt-acc">100%</span>
                                </div>
                                <div class="aisim-stat-row">
                                    <span>${this.t.boundaryNotice}</span>
                                    <span class="aisim-stat-val" id="val-kt-btype">
                                        ${isKnn ? this.str('knn_boundary_knn') : this.str('knn_boundary_tree')}
                                    </span>
                                </div>
                            </div>

                            <div class="aisim-controls" style="margin-top:0.5rem;">
                                <button class="aisim-btn aisim-btn-secondary" id="btn-kt-clear"
                                        style="width:100%; justify-content:center;">
                                    ↺ ${this.str('knn_reset')}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.canvas = this.container.querySelector('#aisim-kt-canvas');
        this.ctx = this.canvas.getContext('2d');
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
    }

    bindEvents() {
        const btnKnn = this.container.querySelector('#btn-tab-knn');
        const btnTree = this.container.querySelector('#btn-tab-tree');
        const btnClsA = this.container.querySelector('#btn-cls-a');
        const btnClsB = this.container.querySelector('#btn-cls-b');
        const sliderK = this.container.querySelector('#slider-k');
        const sliderDepth = this.container.querySelector('#slider-depth');
        const selDataset = this.container.querySelector('#aisim-kt-dataset');
        const btnClear = this.container.querySelector('#btn-kt-clear');

        if (this.readonly) {
            if (btnKnn) {
                btnKnn.disabled = true;
            }
            if (btnTree) {
                btnTree.disabled = true;
            }
            if (btnClsA) {
                btnClsA.disabled = true;
            }
            if (btnClsB) {
                btnClsB.disabled = true;
            }
            if (sliderK) {
                sliderK.disabled = true;
            }
            if (sliderDepth) {
                sliderDepth.disabled = true;
            }
            if (selDataset) {
                selDataset.disabled = true;
            }
            if (btnClear) {
                btnClear.disabled = true;
            }
            return;
        }

        btnKnn.addEventListener('click', () => {
            this.modelType = 'knn';
            this.container.querySelector('#card-param-knn').style.display = 'block';
            this.container.querySelector('#card-param-tree').style.display = 'none';
            btnKnn.classList.add('active');
            btnTree.classList.remove('active');
            this.container.querySelector('#aisim-kt-advice').textContent = this.t.knnNotice;
            this.container.querySelector('#val-kt-btype').textContent = this.str('knn_boundary_knn');
            this.update();
        });

        btnTree.addEventListener('click', () => {
            this.modelType = 'tree';
            this.container.querySelector('#card-param-knn').style.display = 'none';
            this.container.querySelector('#card-param-tree').style.display = 'block';
            btnTree.classList.add('active');
            btnKnn.classList.remove('active');
            this.container.querySelector('#aisim-kt-advice').textContent = this.t.treeNotice;
            this.container.querySelector('#val-kt-btype').textContent = this.str('knn_boundary_tree');
            this.update();
        });

        btnClsA.addEventListener('click', () => {
            this.activeClass = 0;
            btnClsA.className = 'aisim-btn aisim-btn-primary';
            btnClsB.className = 'aisim-btn aisim-btn-secondary';
        });

        btnClsB.addEventListener('click', () => {
            this.activeClass = 1;
            btnClsA.className = 'aisim-btn aisim-btn-secondary';
            btnClsB.className = 'aisim-btn aisim-btn-danger';
        });

        sliderK.addEventListener('input', (e) => {
            this.k = parseInt(e.target.value, 10);
            this.container.querySelector('#val-k').textContent = this.k;
            this.update();
        });

        sliderDepth.addEventListener('input', (e) => {
            this.maxDepth = parseInt(e.target.value, 10);
            this.container.querySelector('#val-depth').textContent = this.maxDepth;
            this.update();
        });

        selDataset.addEventListener('change', (e) => {
            this.datasetKey = e.target.value;
            this.initData();
            this.update();
        });

        btnClear.addEventListener('click', () => {
            this.initData();
            this.update();
        });

        // Canvas point addition.
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const x = -2.0 + (clickX / this.canvasWidth) * 4.0;
            const y = 2.0 - (clickY / this.canvasHeight) * 4.0;
            this.points.push({x, y, cls: this.activeClass});
            this.update();
        });

        // Mouse hover for nearest neighbors visualization.
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.modelType !== 'knn') {
                return;
            }
            const rect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            this.hoverPoint = {
                x: -2.0 + (clickX / this.canvasWidth) * 4.0,
                y: 2.0 - (clickY / this.canvasHeight) * 4.0,
                px: clickX,
                py: clickY
            };
            this.draw();
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hoverPoint = null;
            this.draw();
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

    predictKNN(x, y) {
        // Calculate Euclidean distance to all points.
        const dists = this.points.map(pt => ({
            cls: pt.cls,
            pt,
            d: Math.pow(pt.x - x, 2) + Math.pow(pt.y - y, 2)
        }));
        dists.sort((a, b) => a.d - b.d);

        const neighbors = dists.slice(0, Math.min(this.k, dists.length));
        let count0 = 0;
        let count1 = 0;
        for (const n of neighbors) {
            if (n.cls === 0) {
                count0++;
            } else {
                count1++;
            }
        }
        return {
            pred: count1 > count0 ? 1 : 0,
            neighbors
        };
    }

    buildDecisionTree() {
        // Build axis-aligned decision tree splits recursively.
        const buildNode = (pts, depth) => {
            if (pts.length === 0) {
                return {isLeaf: true, cls: 0};
            }
            const c0 = pts.filter(p => p.cls === 0).length;
            const c1 = pts.length - c0;
            if (depth >= this.maxDepth || c0 === 0 || c1 === 0) {
                return {isLeaf: true, cls: c1 > c0 ? 1 : 0};
            }

            // Find best split on x or y minimizing Gini impurity.
            let bestGini = 1.0;
            let bestAxis = 'x';
            let bestTh = 0;

            for (const axis of ['x', 'y']) {
                const vals = pts.map(p => p[axis]).sort((a, b) => a - b);
                for (let i = 0; i < vals.length - 1; i++) {
                    const th = (vals[i] + vals[i + 1]) / 2;
                    const left = pts.filter(p => p[axis] <= th);
                    const right = pts.filter(p => p[axis] > th);
                    if (left.length === 0 || right.length === 0) {
                        continue;
                    }

                    const giniOf = (group) => {
                        const p0 = group.filter(p => p.cls === 0).length / group.length;
                        const p1 = 1 - p0;
                        return 1 - (p0 * p0 + p1 * p1);
                    };

                    const gini = (left.length / pts.length) * giniOf(left) +
                                 (right.length / pts.length) * giniOf(right);

                    if (gini < bestGini) {
                        bestGini = gini;
                        bestAxis = axis;
                        bestTh = th;
                    }
                }
            }

            if (bestGini >= 0.5) {
                return {isLeaf: true, cls: c1 > c0 ? 1 : 0};
            }

            const leftPts = pts.filter(p => p[bestAxis] <= bestTh);
            const rightPts = pts.filter(p => p[bestAxis] > bestTh);

            return {
                isLeaf: false,
                axis: bestAxis,
                th: bestTh,
                left: buildNode(leftPts, depth + 1),
                right: buildNode(rightPts, depth + 1)
            };
        };

        this.treeRoot = buildNode(this.points, 0);
    }

    predictTree(node, x, y) {
        if (!node || node.isLeaf) {
            return node ? node.cls : 0;
        }
        const val = node.axis === 'x' ? x : y;
        if (val <= node.th) {
            return this.predictTree(node.left, x, y);
        }
        return this.predictTree(node.right, x, y);
    }

    update() {
        if (this.modelType === 'tree') {
            this.buildDecisionTree();
        }

        // Calculate training accuracy.
        let correct = 0;
        for (const pt of this.points) {
            const pred = this.modelType === 'knn'
                ? this.predictKNN(pt.x, pt.y).pred
                : this.predictTree(this.treeRoot, pt.x, pt.y);
            if (pred === pt.cls) {
                correct++;
            }
        }
        const acc = this.points.length > 0 ? Math.round((correct / this.points.length) * 100) : 100;
        this.container.querySelector('#val-kt-acc').textContent = `${acc}%`;

        this.draw();

        if (typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'knn_tree',
                parameters: {
                    model: this.modelType,
                    k: this.k,
                    depth: this.maxDepth,
                    dataset: this.datasetKey
                },
                metrics: {
                    testAcc: Math.round((correct / (this.points.length || 1)) * 1000) / 1000
                }
            });
        }
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;
        ctx.clearRect(0, 0, w, h);

        // Draw decision boundary regions on grid (resolution step 8px for high perf).
        const step = 8;
        for (let px = 0; px < w; px += step) {
            const x = -2.0 + (px / w) * 4.0;
            for (let py = 0; py < h; py += step) {
                const y = 2.0 - (py / h) * 4.0;
                const pred = this.modelType === 'knn'
                    ? this.predictKNN(x, y).pred
                    : this.predictTree(this.treeRoot, x, y);

                ctx.fillStyle = pred === 0 ? 'rgba(37, 99, 235, 0.16)' : 'rgba(239, 68, 68, 0.16)';
                ctx.fillRect(px, py, step, step);
            }
        }

        // Draw coordinate crosshair lines.
        ctx.strokeStyle = 'rgba(203, 213, 225, 0.6)';
        ctx.lineWidth = 1;
        const origin = this.toPx(0, 0);
        ctx.beginPath();
        ctx.moveTo(0, origin.py);
        ctx.lineTo(w, origin.py);
        ctx.moveTo(origin.px, 0);
        ctx.lineTo(origin.px, h);
        ctx.stroke();

        // If hoverPoint exists in k-NN, draw lines to nearest neighbors.
        if (this.hoverPoint && this.modelType === 'knn') {
            const {neighbors, pred} = this.predictKNN(this.hoverPoint.x, this.hoverPoint.y);
            for (const n of neighbors) {
                const nPx = this.toPx(n.pt.x, n.pt.y);
                ctx.strokeStyle = n.cls === 0 ? '#2563eb' : '#ef4444';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(this.hoverPoint.px, this.hoverPoint.py);
                ctx.lineTo(nPx.px, nPx.py);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Draw query point cursor.
            ctx.fillStyle = pred === 0 ? '#2563eb' : '#ef4444';
            ctx.beginPath();
            ctx.arc(this.hoverPoint.px, this.hoverPoint.py, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw points.
        for (const pt of this.points) {
            const {px, py} = this.toPx(pt.x, pt.y);
            ctx.fillStyle = pt.cls === 0 ? '#2563eb' : '#ef4444';
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.8;
            ctx.stroke();
        }
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
KnnTreeSim.STRINGS = [
    'knn_title', 'knn_badge', 'knn_model_label', 'knn_knn_model', 'knn_tree_model', 'knn_dataset_label', 'knn_k_param',
    'knn_depth_param', 'knn_class_apts', 'knn_class_bpts', 'knn_add_point_hint', 'knn_active_class_label',
    'knn_accuracy', 'knn_boundary_notice', 'knn_knn_notice', 'knn_tree_notice', 'knn_datasets_moons',
    'knn_datasets_circles', 'knn_datasets_blobs', 'knn_k_hint', 'knn_depth', 'knn_depth_hint', 'knn_metrics',
    'knn_boundary_knn', 'knn_boundary_tree', 'knn_reset'
];
