/**
 * Reinforcement learning (Q-learning): an agent learns to reach the goal in a small grid world.
 *
 * Episodes use a seeded generator, so the same settings and number of episodes always give the same Q-table
 * (a quiz review re-trains to the saved number of episodes and shows the student's own policy).
 *
 * @module filter_aisim/sims/q_learning
 * @copyright 2026 Boban Blagojevic
 * @license http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

/** Maps: S start, G goal, T small treasure, # wall, C cliff. */
const MAPS = {
    maze: [
        'S..#..',
        '.#.#.#',
        '.#....',
        '.##.#.',
        '...#..',
        '.#...G'
    ],
    cliff: [
        '..........',
        '..........',
        '..........',
        'SCCCCCCCCG'
    ],
    treasure: [
        '.......',
        'T.S...G',
        '.......'
    ]
};

const REWARD = {step: -0.1, goal: 10, treasure: 3, cliff: -10};
/** Up, right, down, left. */
const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];
const ARROWS = ['↑', '→', '↓', '←'];
const MAX_STEPS = 100;
const MAX_EPISODES = 2000;

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

const clamp = (v, lo, hi, fallback) => {
    const n = parseFloat(v);
    return isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

export class QLearningSim {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     */
    constructor(container, options = {}) {
        this.container = container;
        this.strings = options.strings || {};

        this.t = {
            title: this.str('ql_title'),
            badge: this.str('ql_badge'),
            params: this.str('ql_params'),
            alpha: this.str('ql_alpha'),
            gamma: this.str('ql_gamma'),
            epsilon: this.str('ql_epsilon'),
            train: this.str('ql_train'),
            pause: this.str('ql_pause'),
            one: this.str('ql_one'),
            fifty: this.str('ql_fifty'),
            reset: this.str('ql_reset'),
            metrics: this.str('ql_metrics'),
            episode: this.str('ql_episode'),
            avgReward: this.str('ql_avg_reward'),
            path: this.str('ql_path'),
            optimal: this.str('ql_optimal'),
            steps: this.str('ql_steps'),
            noGoal: this.str('ql_no_goal'),
            toTreasure: this.str('ql_to_treasure'),
            rewardCurve: this.str('ql_reward_curve'),
            cell: this.str('ql_cell'),
            cellHint: this.str('ql_cell_hint'),
            legend: this.str('ql_legend'),
            maps: {
                maze: this.str('ql_maps_maze'),
                cliff: this.str('ql_maps_cliff'),
                treasure: this.str('ql_maps_treasure')
            },
            adviceStart: this.str('ql_advice_start'),
            adviceSolved: this.str('ql_advice_solved'),
            adviceLonger: this.str('ql_advice_longer'),
            adviceTreasure: this.str('ql_advice_treasure'),
            adviceTreasureExplore: this.str('ql_advice_treasure_explore'),
            adviceExplore: this.str('ql_advice_explore')
        };

        this.mapKey = Object.prototype.hasOwnProperty.call(MAPS, options.map) ? options.map : 'maze';
        this.alpha = clamp(options.alpha, 0.01, 1, 0.5);
        this.gamma = clamp(options.gamma, 0, 1, 0.9);
        this.epsilon = clamp(options.epsilon, 0, 1, 0.2);
        this.seed = parseInt(options.seed, 10) || 1;
        this.onChange = options.onChange || null;
        this.readonly = !!options.readonly;
        this.running = false;
        this.lastNotify = 0;
        this.selected = null;

        this.initWorld();
        this.reset();
        this.renderLayout();
        this.bindEvents();
        // Re-train to a saved state (quiz review, shared links): same seed and settings give the same Q-table.
        this.trainEpisodes(Math.min(MAX_EPISODES, Math.max(0, parseInt(options.episodes, 10) || 0)));
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

    initWorld() {
        this.grid = MAPS[this.mapKey].map((row) => row.split(''));
        this.rows = this.grid.length;
        this.cols = this.grid[0].length;
        this.grid.forEach((row, r) => row.forEach((ch, c) => {
            if (ch === 'S') {
                this.start = {r, c};
            }
        }));
        this.optimal = this.shortestPath();
    }

    reset() {
        this.Q = Array.from({length: this.rows * this.cols}, () => [0, 0, 0, 0]);
        this.rand = seeded(this.seed * 7919 + Object.keys(MAPS).indexOf(this.mapKey));
        this.episode = 0;
        this.rewards = [];
    }

    cell(r, c) {
        return this.grid[r][c];
    }

    /**
     * One move: walls and the border keep the agent in place.
     *
     * @param {Object} s state {r, c}
     * @param {Number} a action index
     * @returns {Object} {next, reward, done, end} where end is 'goal', 'treasure', 'cliff' or ''
     */
    step(s, a) {
        let r = s.r + MOVES[a][0];
        let c = s.c + MOVES[a][1];
        if (r < 0 || r >= this.rows || c < 0 || c >= this.cols || this.cell(r, c) === '#') {
            r = s.r;
            c = s.c;
        }
        const ch = this.cell(r, c);
        if (ch === 'G') {
            return {next: {r, c}, reward: REWARD.goal, done: true, end: 'goal'};
        }
        if (ch === 'T') {
            return {next: {r, c}, reward: REWARD.treasure, done: true, end: 'treasure'};
        }
        if (ch === 'C') {
            return {next: {r, c}, reward: REWARD.cliff, done: true, end: 'cliff'};
        }
        return {next: {r, c}, reward: REWARD.step, done: false, end: ''};
    }

    /**
     * Best action by the Q-table; ties go to the first one, so an untrained agent always tries "up".
     *
     * @param {Object} s state
     * @returns {Number}
     */
    greedy(s) {
        const q = this.Q[s.r * this.cols + s.c];
        let best = 0;
        for (let a = 1; a < 4; a++) {
            if (q[a] > q[best]) {
                best = a;
            }
        }
        return best;
    }

    /**
     * One training episode from the start: ε-greedy moves and the Q-learning update
     * Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s',·) − Q(s,a)).
     */
    trainEpisode() {
        let s = {...this.start};
        let total = 0;
        for (let i = 0; i < MAX_STEPS; i++) {
            const a = this.rand() < this.epsilon ? Math.floor(this.rand() * 4) : this.greedy(s);
            const {next, reward, done} = this.step(s, a);
            const q = this.Q[s.r * this.cols + s.c];
            const future = done ? 0 : Math.max(...this.Q[next.r * this.cols + next.c]);
            q[a] += this.alpha * (reward + this.gamma * future - q[a]);
            total += reward;
            s = next;
            if (done) {
                break;
            }
        }
        this.episode++;
        this.rewards.push(total);
    }

    trainEpisodes(count) {
        for (let i = 0; i < count && this.episode < MAX_EPISODES; i++) {
            this.trainEpisode();
        }
    }

    /**
     * Follow the greedy policy from the start (no exploration).
     *
     * @returns {Object} {cells, end, steps}
     */
    greedyPath() {
        let s = {...this.start};
        const cells = [s];
        const seen = new Set([`${s.r},${s.c}`]);
        for (let i = 0; i < MAX_STEPS; i++) {
            const {next, done, end} = this.step(s, this.greedy(s));
            cells.push(next);
            if (done) {
                return {cells, end, steps: cells.length - 1};
            }
            const key = `${next.r},${next.c}`;
            if (seen.has(key)) {
                // A loop: the policy never gets anywhere.
                return {cells, end: '', steps: cells.length - 1};
            }
            seen.add(key);
            s = next;
        }
        return {cells, end: '', steps: MAX_STEPS};
    }

    /**
     * Length of the shortest safe path from the start to the goal (breadth-first search).
     *
     * @returns {Number}
     */
    shortestPath() {
        const dist = new Map([[`${this.start.r},${this.start.c}`, 0]]);
        const queue = [this.start];
        while (queue.length) {
            const s = queue.shift();
            const d = dist.get(`${s.r},${s.c}`);
            for (let a = 0; a < 4; a++) {
                const {next, end} = this.step(s, a);
                const key = `${next.r},${next.c}`;
                if (end === 'goal') {
                    return d + 1;
                }
                if (!end && !dist.has(key)) {
                    dist.set(key, d + 1);
                    queue.push(next);
                }
            }
        }
        return 0;
    }

    renderLayout() {
        const opt = (value, label, current) => `<option value="${value}" ${value === current ? 'selected' : ''}>${label}</option>`;
        const slider = (id, label, value, min, max, step) => `
            <div class="aisim-slider-group">
                <label for="aisim-ql-${id}">${label} = <span id="aisim-ql-${id}-val">${value.toFixed(2)}</span></label>
                <input type="range" class="aisim-slider" id="aisim-ql-${id}"
                       min="${min}" max="${max}" step="${step}" value="${value}">
            </div>`;
        this.container.innerHTML = `
            <div class="aisim-container aisim-ql">
                <div class="aisim-header">
                    <div class="aisim-title">
                        <span>${this.t.title}</span>
                        <span class="aisim-badge">${this.t.badge}</span>
                    </div>
                    <select class="aisim-select" id="aisim-ql-map">
                        ${Object.keys(MAPS).map((k) => opt(k, this.t.maps[k], this.mapKey)).join('')}
                    </select>
                </div>

                <div class="aisim-body">
                    <div class="aisim-grid">
                        <div class="aisim-col-left">
                            <div class="aisim-canvas-box" style="padding:0; overflow:hidden; min-height:0;">
                                <canvas class="aisim-canvas" id="aisim-ql-canvas" style="cursor:pointer;"></canvas>
                            </div>
                            <div class="aisim-ql-legend">${this.t.legend}</div>
                            <div class="aisim-card aisim-advice-card">
                                <div class="aisim-advice-icon">💡</div>
                                <div class="aisim-advice-text" id="aisim-ql-advice">${this.t.adviceStart}</div>
                            </div>
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.cell}</div>
                                <div class="aisim-conv-calc" id="aisim-ql-cell">${this.t.cellHint}</div>
                            </div>
                        </div>

                        <div class="aisim-col-right">
                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.params}</div>
                                ${slider('alpha', this.t.alpha, this.alpha, 0.05, 1, 0.05)}
                                ${slider('gamma', this.t.gamma, this.gamma, 0, 1, 0.05)}
                                ${slider('epsilon', this.t.epsilon, this.epsilon, 0, 1, 0.05)}
                                <div class="aisim-controls">
                                    <button class="aisim-btn aisim-btn-primary" id="aisim-ql-run">${this.t.train}</button>
                                    <button class="aisim-btn aisim-btn-secondary" id="aisim-ql-one">${this.t.one}</button>
                                    <button class="aisim-btn aisim-btn-secondary" id="aisim-ql-fifty">${this.t.fifty}</button>
                                    <button class="aisim-btn aisim-btn-danger" id="aisim-ql-reset">${this.t.reset}</button>
                                </div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.metrics}</div>
                                <div class="aisim-stat-row"><span>${this.t.episode}</span>
                                    <span class="aisim-stat-val" id="aisim-ql-episode">0</span></div>
                                <div class="aisim-stat-row"><span>${this.t.avgReward}</span>
                                    <span class="aisim-stat-val" id="aisim-ql-avg">–</span></div>
                                <div class="aisim-stat-row"><span>${this.t.path}</span>
                                    <span class="aisim-stat-val" id="aisim-ql-path">–</span></div>
                                <div class="aisim-stat-row"><span>${this.t.optimal}</span>
                                    <span class="aisim-stat-val" id="aisim-ql-optimal">–</span></div>
                            </div>

                            <div class="aisim-card">
                                <div class="aisim-card-title">${this.t.rewardCurve}</div>
                                <canvas class="aisim-canvas" id="aisim-ql-curve" style="height:110px;"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.canvas = this.container.querySelector('#aisim-ql-canvas');
        this.curve = this.container.querySelector('#aisim-ql-curve');
        this.setupDPI();
    }

    setupDPI() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.getBoundingClientRect().width || 400;
        // Square cells: the height follows the map shape.
        const h = Math.min(w * this.rows / this.cols, 460);
        this.canvas.style.height = `${h}px`;
        this.sizesPx = {};
        for (const [key, cv, ch] of [['main', this.canvas, h], ['curve', this.curve, 110]]) {
            const cw = cv.getBoundingClientRect().width || w;
            cv.width = cw * dpr;
            cv.height = ch * dpr;
            const ctx = cv.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this.sizesPx[key] = {w: cw, h: ch, ctx};
        }
    }

    bindEvents() {
        const q = (id) => this.container.querySelector(id);
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const {cs, ox, oy} = this.layout();
            const c = Math.floor((e.clientX - rect.left - ox) / cs);
            const r = Math.floor((e.clientY - rect.top - oy) / cs);
            if (r >= 0 && r < this.rows && c >= 0 && c < this.cols) {
                this.selected = {r, c};
                this.update(false);
            }
        });
        window.addEventListener('resize', () => {
            this.setupDPI();
            this.draw();
        });
        const controls = ['#aisim-ql-map', '#aisim-ql-alpha', '#aisim-ql-gamma', '#aisim-ql-epsilon', '#aisim-ql-run',
            '#aisim-ql-one', '#aisim-ql-fifty', '#aisim-ql-reset'];
        if (this.readonly) {
            controls.forEach((id) => {
                q(id).disabled = true;
            });
            return;
        }
        // Any change of the world or the parameters starts learning from scratch, so the result can be replayed.
        const rebuild = () => {
            this.stop();
            this.reset();
            this.update(true);
        };
        q('#aisim-ql-map').addEventListener('change', (e) => {
            this.mapKey = e.target.value;
            this.initWorld();
            this.selected = null;
            this.setupDPI();
            rebuild();
        });
        for (const key of ['alpha', 'gamma', 'epsilon']) {
            q(`#aisim-ql-${key}`).addEventListener('input', (e) => {
                this[key] = parseFloat(e.target.value);
                q(`#aisim-ql-${key}-val`).textContent = this[key].toFixed(2);
                rebuild();
            });
        }
        q('#aisim-ql-run').addEventListener('click', () => (this.running ? this.stop() : this.run()));
        q('#aisim-ql-one').addEventListener('click', () => {
            this.stop();
            this.trainEpisodes(1);
            this.update(true);
        });
        q('#aisim-ql-fifty').addEventListener('click', () => {
            this.stop();
            this.trainEpisodes(50);
            this.update(true);
        });
        q('#aisim-ql-reset').addEventListener('click', () => {
            this.seed++;
            rebuild();
        });
    }

    run() {
        this.running = true;
        this.container.querySelector('#aisim-ql-run').textContent = this.t.pause;
        const frame = () => {
            if (!this.running) {
                return;
            }
            this.trainEpisodes(5);
            this.update(false);
            if (this.episode >= MAX_EPISODES) {
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
        this.container.querySelector('#aisim-ql-run').textContent = this.t.train;
        this.update(true);
    }

    /**
     * Refresh numbers and drawings; tell the listener (quiz answer) at most every 250 ms while training.
     *
     * @param {Boolean} notify force the onChange call
     */
    update(notify) {
        const q = (id) => this.container.querySelector(id);
        const path = this.greedyPath();
        this.path = path;
        const last = this.rewards.slice(-20);
        const avg = last.length ? last.reduce((a, v) => a + v, 0) / last.length : null;
        q('#aisim-ql-episode').textContent = this.episode;
        q('#aisim-ql-avg').textContent = avg === null ? '–' : avg.toFixed(2);
        let pathText = this.t.noGoal;
        if (path.end === 'goal') {
            pathText = `${path.steps} ${this.t.steps}`;
        } else if (path.end === 'treasure') {
            pathText = this.t.toTreasure;
        }
        q('#aisim-ql-path').textContent = pathText;
        q('#aisim-ql-optimal').textContent = `${this.optimal} ${this.t.steps}`;

        let advice = this.t.adviceStart;
        if (path.end === 'goal') {
            advice = path.steps <= this.optimal ? this.t.adviceSolved : this.t.adviceLonger;
        } else if (path.end === 'treasure') {
            // Below about γ = 0.6 the far goal is worth less than the near treasure, however well the agent explores.
            advice = this.gamma < 0.6 ? this.t.adviceTreasure : this.t.adviceTreasureExplore;
        } else if (this.episode >= 30) {
            advice = this.t.adviceExplore;
        }
        q('#aisim-ql-advice').textContent = advice;

        this.drawCell();
        this.draw();

        const now = Date.now();
        if (typeof this.onChange === 'function' && (notify || now - this.lastNotify > 250)) {
            this.lastNotify = now;
            this.onChange({
                simtype: 'q_learning',
                parameters: {
                    map: this.mapKey,
                    alpha: this.alpha,
                    gamma: this.gamma,
                    epsilon: this.epsilon,
                    seed: this.seed,
                    episodes: this.episode
                },
                metrics: {
                    reached: path.end === 'goal',
                    steps: path.end === 'goal' ? path.steps : 0,
                    optimal: this.optimal,
                    avgReward: avg === null ? 0 : Math.round(avg * 100) / 100
                }
            });
        }
    }

    drawCell() {
        const box = this.container.querySelector('#aisim-ql-cell');
        if (!this.selected) {
            box.textContent = this.t.cellHint;
            return;
        }
        const {r, c} = this.selected;
        const qv = this.Q[r * this.cols + c];
        const best = this.greedy(this.selected);
        box.innerHTML = `<div><strong>${this.str('ql_cell_name')} (${r + 1}, ${c + 1})</strong></div>` +
            qv.map((v, a) => {
                const star = a === best && v !== 0 ? ' ★' : '';
                return `<div class="aisim-conv-step">${ARROWS[a]} Q = ${v.toFixed(3)}${star}</div>`;
            }).join('');
    }

    /**
     * Cell size and offsets that centre the map in the canvas.
     *
     * @returns {Object} {cs, ox, oy}
     */
    layout() {
        const {w, h} = this.sizesPx.main;
        const cs = Math.min(w / this.cols, h / this.rows);
        return {cs, ox: (w - cs * this.cols) / 2, oy: (h - cs * this.rows) / 2};
    }

    draw() {
        this.drawGrid();
        this.drawCurve();
    }

    drawGrid() {
        const {w, h, ctx} = this.sizesPx.main;
        const {cs, ox, oy} = this.layout();
        ctx.clearRect(0, 0, w, h);
        const values = this.Q.map((q) => Math.max(...q));
        const maxV = Math.max(0.001, ...values.map((v) => Math.abs(v)));
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const ch = this.cell(r, c);
                const x = ox + c * cs;
                const y = oy + r * cs;
                const v = values[r * this.cols + c];
                let fill = '#f8fafc';
                if (ch === '#') {
                    fill = '#334155';
                } else if (ch === 'C') {
                    fill = '#fca5a5';
                } else if (ch === 'G') {
                    fill = '#86efac';
                } else if (ch === 'T') {
                    fill = '#fde68a';
                } else if (v > 0) {
                    fill = `rgba(34, 112, 122, ${Math.min(0.75, 0.08 + 0.67 * v / maxV)})`;
                } else if (v < 0) {
                    fill = `rgba(239, 68, 68, ${Math.min(0.35, 0.05 + 0.3 * -v / maxV)})`;
                }
                ctx.fillStyle = fill;
                ctx.fillRect(x, y, cs, cs);
                ctx.strokeStyle = '#cbd5e1';
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                if ('SGT'.includes(ch)) {
                    ctx.fillStyle = '#0f172a';
                    ctx.font = `bold ${Math.round(cs * 0.38)}px sans-serif`;
                    ctx.fillText(ch, x + cs / 2, y + cs / 2);
                } else if (ch === '.' && this.Q[r * this.cols + c].some((qv) => qv !== 0)) {
                    ctx.fillStyle = 'rgba(15, 23, 42, 0.55)';
                    ctx.font = `${Math.round(cs * 0.34)}px sans-serif`;
                    ctx.fillText(ARROWS[this.greedy({r, c})], x + cs / 2, y + cs / 2);
                }
            }
        }
        // Greedy path from the start.
        const pts = this.path ? this.path.cells : [];
        if (pts.length > 1) {
            ctx.strokeStyle = this.path.end === 'goal' ? '#f59e0b' : '#ef4444';
            ctx.lineWidth = Math.max(2, cs * 0.09);
            ctx.lineJoin = 'round';
            ctx.beginPath();
            pts.forEach((p, i) => {
                const px = ox + (p.c + 0.5) * cs;
                const py = oy + (p.r + 0.5) * cs;
                if (i === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            });
            ctx.stroke();
        }
        if (this.selected) {
            ctx.strokeStyle = '#7c3aed';
            ctx.lineWidth = 3;
            ctx.strokeRect(ox + this.selected.c * cs + 1.5, oy + this.selected.r * cs + 1.5, cs - 3, cs - 3);
        }
    }

    drawCurve() {
        const {w, h, ctx} = this.sizesPx.curve;
        ctx.clearRect(0, 0, w, h);
        const data = this.rewards;
        if (data.length < 2) {
            return;
        }
        const lo = Math.min(...data, 0);
        const hi = Math.max(...data, 1);
        const toY = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 16);
        const toX = (i) => 6 + (i / (data.length - 1)) * (w - 12);
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, toY(0));
        ctx.lineTo(w, toY(0));
        ctx.stroke();
        ctx.fillStyle = 'rgba(124, 58, 237, 0.25)';
        const stride = Math.max(1, Math.floor(data.length / 400));
        for (let i = 0; i < data.length; i += stride) {
            ctx.fillRect(toX(i) - 1, toY(data[i]) - 1, 2, 2);
        }
        // Moving average over 20 episodes.
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let sum = 0;
        data.forEach((v, i) => {
            sum += v - (i >= 20 ? data[i - 20] : 0);
            const y = toY(sum / Math.min(i + 1, 20));
            if (i === 0) {
                ctx.moveTo(toX(i), y);
            } else {
                ctx.lineTo(toX(i), y);
            }
        });
        ctx.stroke();
    }
}

/** Identifiers of the filter_aisim language strings the simulation uses. */
QLearningSim.STRINGS = [
    'ql_title', 'ql_badge', 'ql_params', 'ql_alpha', 'ql_gamma', 'ql_epsilon', 'ql_train', 'ql_pause', 'ql_one',
    'ql_fifty', 'ql_reset', 'ql_metrics', 'ql_episode', 'ql_avg_reward', 'ql_path', 'ql_optimal', 'ql_steps',
    'ql_no_goal', 'ql_to_treasure', 'ql_reward_curve', 'ql_cell', 'ql_cell_hint', 'ql_legend', 'ql_maps_maze',
    'ql_maps_cliff', 'ql_maps_treasure', 'ql_advice_start', 'ql_advice_solved', 'ql_advice_longer',
    'ql_advice_treasure', 'ql_advice_treasure_explore', 'ql_advice_explore', 'ql_cell_name'
];
