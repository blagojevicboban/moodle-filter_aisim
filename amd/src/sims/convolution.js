/**
 * Convolution & image filters: a 3×3 kernel slides over a small grayscale image.
 *
 * The student picks a ready-made filter or types the kernel weights, sees the filtered image and, by tapping a pixel,
 * the calculation for that pixel. With a target filter the output is compared with the target image (R²).
 *
 * @module filter_aisim/sims/convolution
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

const SIZE = 28;

/** Ready-made kernels. Edge detectors show the absolute value (an edge can be dark→light or light→dark). */
const PRESETS = {
    identity: {kernel: [0, 0, 0, 0, 1, 0, 0, 0, 0], divisor: 1, abs: false},
    box: {kernel: [1, 1, 1, 1, 1, 1, 1, 1, 1], divisor: 9, abs: false},
    gaussian: {kernel: [1, 2, 1, 2, 4, 2, 1, 2, 1], divisor: 16, abs: false},
    sharpen: {kernel: [0, -1, 0, -1, 5, -1, 0, -1, 0], divisor: 1, abs: false},
    sobelx: {kernel: [-1, 0, 1, -2, 0, 2, -1, 0, 1], divisor: 1, abs: true},
    sobely: {kernel: [-1, -2, -1, 0, 0, 0, 1, 2, 1], divisor: 1, abs: true},
    laplacian: {kernel: [0, 1, 0, 1, -4, 1, 0, 1, 0], divisor: 1, abs: true},
    emboss: {kernel: [-2, -1, 0, -1, 1, 1, 0, 1, 2], divisor: 1, abs: false}
};

const IMAGES = ['shapes', 'stripes', 'face', 'noisy'];

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
 * Parse a kernel given as an array or as a comma separated string; null if it is not 9 numbers.
 *
 * @param {Array|String} value
 * @returns {Array|null}
 */
const parseKernel = (value) => {
    const list = Array.isArray(value) ? value : String(value || '').split(',');
    const nums = list.map((v) => Math.max(-9, Math.min(9, Math.round(parseFloat(v)))));
    return nums.length === 9 && nums.every((v) => isFinite(v)) ? nums : null;
};

export class ConvolutionSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('conv_title'),
            badge: this.str('conv_badge'),
            input: this.str('conv_input'),
            output: this.str('conv_output'),
            target: this.str('conv_target'),
            kernel: this.str('conv_kernel'),
            preset: this.str('conv_preset'),
            custom: this.str('conv_custom'),
            divisor: this.str('conv_divisor'),
            abs: this.str('conv_abs'),
            targetSelect: this.str('conv_target_select'),
            noTarget: this.str('conv_no_target'),
            metrics: this.str('conv_metrics'),
            sum: this.str('conv_sum'),
            match: this.str('conv_match'),
            calc: this.str('conv_calc'),
            calcHint: this.str('conv_calc_hint'),
            images: {
                shapes: this.str('conv_images_shapes'),
                stripes: this.str('conv_images_stripes'),
                face: this.str('conv_images_face'),
                noisy: this.str('conv_images_noisy')
            },
            presets: {
                identity: this.str('conv_presets_identity'),
                box: this.str('conv_presets_box'),
                gaussian: this.str('conv_presets_gaussian'),
                sharpen: this.str('conv_presets_sharpen'),
                sobelx: this.str('conv_presets_sobelx'),
                sobely: this.str('conv_presets_sobely'),
                laplacian: this.str('conv_presets_laplacian'),
                emboss: this.str('conv_presets_emboss')
            },
            adviceFree: this.str('conv_advice_free'),
            adviceFar: this.str('conv_advice_far'),
            adviceClose: this.str('conv_advice_close'),
            adviceSolved: this.str('conv_advice_solved'),
            adviceBright: this.str('conv_advice_bright'),
            adviceAbs: this.str('conv_advice_abs')
        };

        this.imageKey = IMAGES.includes(options.image) ? options.image : 'shapes';
        // A target given in the options (quiz, challenge) is fixed; without one the student can pick one.
        this.targetFixed = Object.prototype.hasOwnProperty.call(PRESETS, options.target);
        this.targetKey = this.targetFixed ? options.target : '';
        this.showPresets = options.presets !== false && options.presets !== 'false' && options.presets !== '0';
        const start = PRESETS[options.preset] || PRESETS.identity;
        this.kernel = parseKernel(options.kernel) || start.kernel.slice();
        this.divisor = Math.max(1, Math.min(32, parseInt(options.divisor, 10) || (options.kernel ? 1 : start.divisor)));
        if (options.abs === undefined || options.abs === '') {
            this.abs = options.kernel ? false : start.abs;
        } else {
            this.abs = options.abs === true || options.abs === 'true' || options.abs === '1' || options.abs === 1;
        }
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;
        this.selected = {r: 9, c: 9};

        this.initImage();
        this.renderLayout();
        this.bindEvents();
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
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

    initImage() {
        const img = new Float32Array(SIZE * SIZE);
        const set = (fn) => {
            for (let r = 0; r < SIZE; r++) {
                for (let c = 0; c < SIZE; c++) {
                    const v = fn(r, c);
                    if (v !== null) {
                        img[r * SIZE + c] = v;
                    }
                }
            }
        };
        const shapes = () => {
            set(() => 0.12);
            set((r, c) => (r >= 3 && r <= 11 && c >= 3 && c <= 11 ? 0.92 : null));
            set((r, c) => ((r - 8) ** 2 + (c - 20) ** 2 <= 22 ? 0.65 : null));
            set((r, c) => (r >= 15 && r <= 24 && Math.abs(c - 14) <= (r - 15) * 1.2 ? 0.8 : null));
        };
        if (this.imageKey === 'stripes') {
            set((r, c) => {
                if (c < 14) {
                    return Math.floor(c / 3) % 2 ? 0.9 : 0.1;
                }
                return Math.floor(r / 3) % 2 ? 0.9 : 0.1;
            });
        } else if (this.imageKey === 'face') {
            set((r, c) => {
                const d = Math.hypot(r - 13.5, c - 13.5);
                if (d > 12.5) {
                    return 0.08;
                }
                if (d > 11) {
                    return 0.15;
                }
                return 0.85;
            });
            set((r, c) => (Math.hypot(r - 10, c - 9) <= 1.8 || Math.hypot(r - 10, c - 18) <= 1.8 ? 0.1 : null));
            set((r, c) => {
                const d = Math.hypot(r - 14, c - 13.5);
                return d >= 6 && d <= 7.6 && r > 16 ? 0.1 : null;
            });
        } else {
            shapes();
            if (this.imageKey === 'noisy') {
                const rand = seeded(2026);
                for (let i = 0; i < img.length; i++) {
                    const u = Math.max(rand(), 1e-9);
                    const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
                    img[i] = Math.min(1, Math.max(0, img[i] + g * 0.18));
                }
            }
        }
        this.image = img;
        this.targetImage = this.targetKey ? this.apply(PRESETS[this.targetKey]) : null;
    }

    /**
     * Pixel value; outside the image the nearest edge pixel is used (so a blur does not darken the border).
     *
     * @param {Number} r
     * @param {Number} c
     * @returns {Number}
     */
    pixel(r, c) {
        const rr = Math.min(SIZE - 1, Math.max(0, r));
        const cc = Math.min(SIZE - 1, Math.max(0, c));
        return this.image[rr * SIZE + cc];
    }

    /**
     * Raw convolution value (before |x| and clipping) at one pixel.
     *
     * @param {Object} f filter {kernel, divisor}
     * @param {Number} r
     * @param {Number} c
     * @returns {Number}
     */
    rawAt(f, r, c) {
        let s = 0;
        for (let i = 0; i < 9; i++) {
            s += f.kernel[i] * this.pixel(r + Math.floor(i / 3) - 1, c + (i % 3) - 1);
        }
        return s / f.divisor;
    }

    /**
     * Shown value: optional absolute value, then clipped to the 0..1 brightness range.
     *
     * @param {Object} f filter {kernel, divisor, abs}
     * @param {Number} raw
     * @returns {Number}
     */
    shown(f, raw) {
        return Math.min(1, Math.max(0, f.abs ? Math.abs(raw) : raw));
    }

    apply(f) {
        const out = new Float32Array(SIZE * SIZE);
        for (let r = 0; r < SIZE; r++) {
            for (let c = 0; c < SIZE; c++) {
                out[r * SIZE + c] = this.shown(f, this.rawAt(f, r, c));
            }
        }
        return out;
    }

    filter() {
        return {kernel: this.kernel, divisor: this.divisor, abs: this.abs};
    }

    /**
     * Coefficient of determination of the output against the target, clipped to 0..1.
     *
     * @returns {Number|null}
     */
    match() {
        if (!this.targetImage) {
            return null;
        }
        const t = this.targetImage;
        const mean = t.reduce((a, v) => a + v, 0) / t.length;
        let sse = 0;
        let sst = 0;
        for (let i = 0; i < t.length; i++) {
            sse += (this.output[i] - t[i]) ** 2;
            sst += (t[i] - mean) ** 2;
        }
        return Math.max(0, Math.min(1, 1 - sse / Math.max(sst, 1e-9)));
    }

    currentPreset() {
        const found = Object.entries(PRESETS).find(([, p]) => p.divisor === this.divisor && p.abs === this.abs &&
            p.kernel.every((v, i) => v === this.kernel[i]));
        return found ? found[0] : '';
    }

    renderLayout() {
        const opt = (value, label, current) => `<option value="${value}" ${value === current ? 'selected' : ''}>${label}</option>`;
        const presetOptions = [opt('', this.t.custom, this.currentPreset())]
            .concat(Object.keys(PRESETS).map((k) => opt(k, this.t.presets[k], this.currentPreset()))).join('');
        const targetOptions = [opt('', this.t.noTarget, this.targetKey)]
            .concat(Object.keys(PRESETS).filter((k) => k !== 'identity')
                .map((k) => opt(k, this.t.presets[k], this.targetKey))).join('');
        const cells = this.kernel.map((v, i) => `<input type="number" class="aisim-conv-cell" data-i="${i}"
            min="-9" max="9" step="1" value="${v}" inputmode="numeric" aria-label="w${Math.floor(i / 3) + 1}${(i % 3) + 1}">`)
            .join('');
        const imageBox = (id, label) => `
            <div class="aisim-conv-imgbox">
                <div class="aisim-card-title">${label}</div>
                <canvas class="aisim-canvas aisim-conv-canvas" id="${id}"></canvas>
            </div>`;
        this.container.innerHTML = `
            <div class="aisim-container aisim-conv">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <select class="aisim-select" id="aisim-conv-image">
                        ${IMAGES.map((k) => opt(k, this.t.images[k], this.imageKey)).join('')}
                    </select>
                </div>

                <div class="aisim-body">
                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-conv-images">
                                ${imageBox('aisim-conv-in', this.t.input)}
                                ${imageBox('aisim-conv-out', this.t.output)}
                                ${this.targetImage ? imageBox('aisim-conv-target', this.t.target) : ''}
                            </div>
                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-conv-advice">${this.t.adviceFree}</div>
                            </div>
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.calc}</div>
                                <div class="aisim-conv-calc" id="aisim-conv-calc">${this.t.calcHint}</div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.kernel}</div>
                                <div class="aisim-slider-group" ${this.showPresets ? '' : 'style="display:none;"'}>
                                    <label for="aisim-conv-preset">${this.t.preset}</label>
                                    <select class="aisim-select" id="aisim-conv-preset" style="width:100%;">
                                        ${presetOptions}
                                    </select>
                                </div>
                                <div class="aisim-conv-kernel">${cells}</div>
                                <div class="aisim-conv-row">
                                    <label for="aisim-conv-divisor">${this.t.divisor} ÷</label>
                                    <input type="number" class="aisim-conv-cell" id="aisim-conv-divisor"
                                           min="1" max="32" step="1" value="${this.divisor}" inputmode="numeric">
                                </div>
                                <label class="aisim-conv-row aisim-conv-check">
                                    <input type="checkbox" id="aisim-conv-abs" ${this.abs ? 'checked' : ''}>
                                    <span>${this.t.abs}</span>
                                </label>
                            </div>

                            <div class="aisim-card" ${this.targetFixed ? 'style="display:none;"' : ''}>
                                <div class="aisim-slider-group">
                                    <label for="aisim-conv-targetsel">${this.t.targetSelect}</label>
                                    <select class="aisim-select" id="aisim-conv-targetsel" style="width:100%;">
                                        ${targetOptions}
                                    </select>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.metrics}</div>
                                <div class="aisim-stat-row"><span>${this.t.sum}</span>
                                    <span class="aisim-stat-val" id="aisim-conv-sum">1</span></div>
                                <div class="aisim-stat-row" id="aisim-conv-match-row"><span>${this.t.match}</span>
                                    <span class="aisim-stat-val" id="aisim-conv-match">–</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.canvases = {
            input: this.container.querySelector('#aisim-conv-in'),
            output: this.container.querySelector('#aisim-conv-out'),
            target: this.container.querySelector('#aisim-conv-target')
        };
        this.container.querySelector('#aisim-conv-match-row').style.display = this.targetImage ? 'flex' : 'none';
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        this.sizesPx = {};
        for (const [key, cv] of Object.entries(this.canvases)) {
            if (!cv) {
                continue;
            }
            const w = cv.getBoundingClientRect().width || 200;
            cv.width = w * dpr;
            cv.height = w * dpr;
            cv.style.height = `${w}px`;
            const ctx = cv.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this.sizesPx[key] = {w, ctx};
        }
    }

    bindEvents() {
        const q = (sel) => this.container.querySelector(sel);
        const cells = Array.from(this.container.querySelectorAll('.aisim-conv-kernel .aisim-conv-cell'));
        const controls = [q('#aisim-conv-image'), q('#aisim-conv-preset'), q('#aisim-conv-divisor'), q('#aisim-conv-abs'),
            q('#aisim-conv-targetsel')].concat(cells);
        // Pixel inspection stays available in a review.
        for (const key of ['input', 'output', 'target']) {
            const cv = this.canvases[key];
            if (cv) {
                cv.addEventListener('click', (e) => {
                    const rect = cv.getBoundingClientRect();
                    const c = Math.floor(((e.clientX - rect.left) / rect.width) * SIZE);
                    const r = Math.floor(((e.clientY - rect.top) / rect.height) * SIZE);
                    this.selected = {r: Math.min(SIZE - 1, Math.max(0, r)), c: Math.min(SIZE - 1, Math.max(0, c))};
                    this.update(false);
                });
            }
        }
        if (this.readonly) {
            controls.forEach((el) => {
                el.disabled = true;
            });
            return;
        }
        // A fixed target (quiz, challenge) belongs to its image.
        q('#aisim-conv-image').disabled = this.targetFixed;
        const syncInputs = () => {
            cells.forEach((el, i) => {
                el.value = this.kernel[i];
            });
            q('#aisim-conv-divisor').value = this.divisor;
            q('#aisim-conv-abs').checked = this.abs;
        };
        q('#aisim-conv-image').addEventListener('change', (e) => {
            this.imageKey = e.target.value;
            this.initImage();
            this.update();
        });
        q('#aisim-conv-preset').addEventListener('change', (e) => {
            const p = PRESETS[e.target.value];
            if (p) {
                this.kernel = p.kernel.slice();
                this.divisor = p.divisor;
                this.abs = p.abs;
                syncInputs();
                this.update();
            }
        });
        cells.forEach((el) => el.addEventListener('input', () => {
            const v = parseInt(el.value, 10);
            if (isFinite(v)) {
                this.kernel[parseInt(el.dataset.i, 10)] = Math.max(-9, Math.min(9, v));
                this.update();
            }
        }));
        q('#aisim-conv-divisor').addEventListener('input', (e) => {
            const v = parseInt(e.target.value, 10);
            if (isFinite(v) && v >= 1) {
                this.divisor = Math.min(32, v);
                this.update();
            }
        });
        q('#aisim-conv-abs').addEventListener('change', (e) => {
            this.abs = e.target.checked;
            this.update();
        });
        q('#aisim-conv-targetsel').addEventListener('change', (e) => {
            this.targetKey = e.target.value;
            this.initImage();
            // The target box appears or disappears, so the layout is drawn again.
            this.renderLayout();
            this.bindEvents();
            this.update();
        });
    }

    /**
     * Recompute the output, numbers and drawings, and tell the listener (quiz answer).
     *
     * @param {Boolean} notify call onChange (false for a pixel selection only)
     */
    update(notify = true) {
        this.output = this.apply(this.filter());
        const match = this.match();
        const q = (sel) => this.container.querySelector(sel);
        const sum = this.kernel.reduce((a, v) => a + v, 0);
        const ratio = sum / this.divisor;
        q('#aisim-conv-sum').textContent = `${sum} / ${this.divisor} = ${Math.round(ratio * 100) / 100}`;
        q('#aisim-conv-match').textContent = match === null ? '–' : `${Math.round(match * 100)}%`;
        const preset = q('#aisim-conv-preset');
        if (preset) {
            preset.value = this.currentPreset();
        }

        let advice = this.t.adviceFree;
        const target = this.targetKey ? PRESETS[this.targetKey] : null;
        if (match !== null && match >= 0.98) {
            advice = this.t.adviceSolved;
        } else if (target && target.abs && !this.abs && sum === 0) {
            advice = this.t.adviceAbs;
        } else if (sum !== 0 && Math.abs(ratio - 1) > 0.15 && !this.abs) {
            advice = this.t.adviceBright;
        } else if (match !== null) {
            advice = match >= 0.8 ? this.t.adviceClose : this.t.adviceFar;
        }
        q('#aisim-conv-advice').textContent = advice;

        this.drawCalc();
        this.draw();

        if (notify && typeof this.onChange === 'function') {
            this.onChange({
                simtype: 'convolution',
                parameters: {
                    image: this.imageKey,
                    target: this.targetKey,
                    kernel: this.kernel.slice(),
                    divisor: this.divisor,
                    abs: this.abs
                },
                metrics: {
                    match: match === null ? null : Math.round(match * 1000) / 1000
                }
            });
        }
    }

    drawCalc() {
        const {r, c} = this.selected;
        const f = this.filter();
        const fmt = (v) => (Math.round(v * 100) / 100).toString();
        const terms = [];
        for (let i = 0; i < 9; i++) {
            terms.push(`<span>${fmt(this.pixel(r + Math.floor(i / 3) - 1, c + (i % 3) - 1))}×${this.kernel[i]}</span>`);
        }
        const raw = this.rawAt(f, r, c);
        const shown = this.shown(f, raw);
        const steps = [`(${terms.join(' + ')}) ÷ ${this.divisor} = ${fmt(raw)}`];
        if (this.abs) {
            steps.push(`|${fmt(raw)}| = ${fmt(Math.abs(raw))}`);
        }
        if (shown !== (this.abs ? Math.abs(raw) : raw)) {
            steps.push(this.str('conv_clipped', fmt(shown)));
        }
        this.container.querySelector('#aisim-conv-calc').innerHTML =
            `<div><strong>${this.str('conv_pixel')} (${r + 1}, ${c + 1})</strong></div>` +
            steps.map((s) => `<div class="aisim-conv-step">${s}</div>`).join('');
    }

    draw() {
        this.drawImage('input', this.image, true);
        this.drawImage('output', this.output, false);
        if (this.targetImage) {
            this.drawImage('target', this.targetImage, false);
        }
    }

    /**
     * Draw a grayscale image with square pixels, and mark the selected pixel (and its 3×3 window on the input).
     *
     * @param {String} key canvas key
     * @param {Float32Array} data values 0..1
     * @param {Boolean} mark mark the 3×3 window
     */
    drawImage(key, data, mark) {
        const box = this.sizesPx[key];
        if (!box || !data) {
            return;
        }
        const {w, ctx} = box;
        const px = w / SIZE;
        for (let r = 0; r < SIZE; r++) {
            for (let c = 0; c < SIZE; c++) {
                const g = Math.round(data[r * SIZE + c] * 255);
                ctx.fillStyle = `rgb(${g}, ${g}, ${g})`;
                // A little overlap avoids hairline gaps between pixels on fractional sizes.
                ctx.fillRect(c * px, r * px, px + 0.5, px + 0.5);
            }
        }
        const {r, c} = this.selected;
        ctx.lineWidth = 2;
        if (mark) {
            ctx.strokeStyle = '#f59e0b';
            ctx.strokeRect((c - 1) * px, (r - 1) * px, px * 3, px * 3);
        }
        ctx.strokeStyle = '#ef4444';
        ctx.strokeRect(c * px, r * px, px, px);
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
ConvolutionSim.STRINGS = [
    'conv_title', 'conv_badge', 'conv_input', 'conv_output', 'conv_target', 'conv_kernel', 'conv_preset', 'conv_custom',
    'conv_divisor', 'conv_abs', 'conv_target_select', 'conv_no_target', 'conv_metrics', 'conv_sum', 'conv_match',
    'conv_calc', 'conv_calc_hint', 'conv_images_shapes', 'conv_images_stripes', 'conv_images_face', 'conv_images_noisy',
    'conv_presets_identity', 'conv_presets_box', 'conv_presets_gaussian', 'conv_presets_sharpen', 'conv_presets_sobelx',
    'conv_presets_sobely', 'conv_presets_laplacian', 'conv_presets_emboss', 'conv_advice_free', 'conv_advice_far',
    'conv_advice_close', 'conv_advice_solved', 'conv_advice_bright', 'conv_advice_abs', 'conv_clipped', 'conv_pixel'
];
