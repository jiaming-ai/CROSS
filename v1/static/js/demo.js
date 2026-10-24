/* Renders the CROSS vs. commit-on-retrieval toy simulation (sim.js). */
(function () {
  'use strict';
  const root = document.getElementById('demo');
  if (!root || !window.CrossSim) return;

  const $ = (id) => document.getElementById(id);
  const cvG = $('greedyCanvas'), cvC = $('crossCanvas');
  const playBtn = $('demoPlay'), scrub = $('demoScrub');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const { WORLD } = window.CrossSim;

  let sim = null, t = 0, playing = false, speed = 1, lastTs = 0, acc = 0;
  let colors = {};
  const STEPS_PER_SEC = 15;

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    colors = {
      bg: v('--canvas-bg'), corridor: v('--corridor'), ink: v('--ink'), ink2: v('--ink-2'), ink3: v('--ink-3'),
      blue: v('--blue'), green: v('--green'), accent: v('--accent'), red: v('--red'), surface: v('--surface'), good: v('--good'),
    };
  }

  function alpha(hex, a) {
    // hex (#rrggbb) -> rgba
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // ------------------------------------------------------------ canvas setup
  function setup(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = w * WORLD.h / WORLD.w;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, s: w / WORLD.w, w, h };
  }
  let G = null, C = null;
  const P = (S, x, y) => [x * S.s, (WORLD.h - y) * S.s];

  function drawMap(S, hl) {
    const { ctx, s, w, h } = S;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, w, h);
    // corridors
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = colors.corridor; ctx.lineWidth = 1.5 * s;
    sim.map.corridors.forEach((pl) => {
      ctx.beginPath();
      pl.forEach(([x, y], i) => { const [px, py] = P(S, x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    });
    // crowd zone
    const [cx0, cy0] = P(S, sim.CROWD.x0, 4.5 + 0.75), [cx1, cy1] = P(S, sim.CROWD.x1, 4.5 - 0.75);
    ctx.save();
    ctx.beginPath(); ctx.rect(cx0, cy0, cx1 - cx0, cy1 - cy0); ctx.clip();
    ctx.strokeStyle = alpha(colors.ink3, 0.35); ctx.lineWidth = 1;
    for (let x = cx0 - (cy1 - cy0); x < cx1; x += 5) { ctx.beginPath(); ctx.moveTo(x, cy1); ctx.lineTo(x + (cy1 - cy0), cy0); ctx.stroke(); }
    ctx.restore();
    ctx.fillStyle = colors.ink3;
    ctx.font = `500 ${Math.max(9, s * 0.42)}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('crowd: no matches', (cx0 + cx1) / 2, cy1 + 3);
    ctx.textBaseline = 'middle';
    const [lx, ly] = P(S, 10, (4.5 + 10) / 2);
    ctx.fillText('two look-alike corridors', lx, ly);
    // edges + nodes
    ctx.strokeStyle = alpha(colors.ink3, 0.45); ctx.lineWidth = 1;
    ctx.beginPath();
    sim.map.edges.forEach(([a, b]) => {
      const A = sim.map.nodes[a], B = sim.map.nodes[b];
      const [ax, ay] = P(S, A.x, A.y), [bx, by] = P(S, B.x, B.y);
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    });
    ctx.stroke();
    ctx.fillStyle = colors.ink3;
    sim.map.nodes.forEach((n) => {
      const [x, y] = P(S, n.x, n.y);
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
    });
    if (hl) hl.forEach(({ i, color, a }) => {
      const n = sim.map.nodes[i]; const [x, y] = P(S, n.x, n.y);
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = alpha(color, 0.25 * a); ctx.fill();
      ctx.lineWidth = 1.6; ctx.strokeStyle = alpha(color, a); ctx.stroke();
    });
  }

  function drawRobot(S, pose) {
    const { ctx, s } = S;
    const [x, y] = P(S, pose[0], pose[1]);
    const r = Math.max(6, s * 0.38);
    ctx.save();
    ctx.translate(x, y); ctx.rotate(-pose[2]);
    ctx.beginPath(); ctx.moveTo(r, 0); ctx.lineTo(-r * 0.7, r * 0.62); ctx.lineTo(-r * 0.35, 0); ctx.lineTo(-r * 0.7, -r * 0.62); ctx.closePath();
    ctx.fillStyle = colors.ink; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = colors.surface; ctx.stroke(); ctx.fill();
    ctx.restore();
  }

  function ellipse(S, mu, Pc, color, fillA, strokeA, lw) {
    const { ctx, s } = S;
    const a = Pc[0], b = Pc[1], d = Pc[2];
    const tr = (a + d) / 2, det = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr - det));
    const l1 = tr + disc, l2 = Math.max(1e-6, tr - disc);
    const ang = Math.atan2(l1 - a, b || 1e-9);
    const [x, y] = P(S, mu[0], mu[1]);
    ctx.beginPath();
    ctx.ellipse(x, y, Math.max(3, 2 * Math.sqrt(l1) * s), Math.max(3, 2 * Math.sqrt(l2) * s), -ang, 0, Math.PI * 2);
    ctx.fillStyle = alpha(color, fillA); ctx.fill();
    ctx.lineWidth = lw || 1.5; ctx.strokeStyle = alpha(color, strokeA); ctx.stroke();
  }

  function cross(S, x, y, color, a) {
    const { ctx } = S; const [px, py] = P(S, x, y); const r = 5;
    ctx.strokeStyle = alpha(color, a); ctx.lineWidth = 2.2; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r); ctx.moveTo(px + r, py - r); ctx.lineTo(px - r, py + r); ctx.stroke();
  }

  function recentRetrieval(tt) {
    for (let k = tt; k >= Math.max(0, tt - 2); k--) if (sim.retr[k] && sim.retr[k].length) return k;
    return -1;
  }

  // ------------------------------------------------------------ greedy panel
  function drawGreedy() {
    const S = G, { ctx, s } = S;
    const tr = recentRetrieval(t);
    const ge = sim.greedy[tr >= 0 ? tr : 0];
    const hl = [];
    if (tr >= 0 && ge.event) hl.push({ i: ge.event.kf, color: ge.event.kind === 'wrong' ? colors.red : colors.green, a: 1 - (t - tr) / 3 });
    drawMap(S, hl);
    // corrupted memory: false loop-closure edges accumulated so far
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1.6;
    sim.falseEdges.forEach((e) => {
      if (e.t > t) return;
      const fresh = Math.max(0, 1 - (t - e.t) / 12);
      const [ax, ay] = P(S, e.a[0], e.a[1]), [bx, by] = P(S, e.b[0], e.b[1]);
      ctx.strokeStyle = alpha(colors.red, 0.35 + 0.5 * fresh);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    });
    ctx.setLineDash([]);
    // committed estimate trajectory, broken at jumps
    ctx.strokeStyle = colors.accent; ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath();
    let prev = null;
    for (let k = 0; k <= t; k++) {
      const e = sim.greedy[k].est;
      if (!e) continue;
      const [x, y] = P(S, e[0], e[1]);
      if (!prev || Math.hypot(e[0] - prev[0], e[1] - prev[1]) > 1.2) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      prev = e;
    }
    ctx.stroke();
    // lift line for the chosen match
    const cur = sim.greedy[t];
    if (tr >= 0 && ge.event && cur.est) {
      const n = sim.map.nodes[ge.event.kf];
      const [nx, ny] = P(S, n.x, n.y), [ex, ey] = P(S, ge.est[0], ge.est[1]);
      ctx.setLineDash([2, 3]); ctx.lineWidth = 1.2;
      ctx.strokeStyle = alpha(ge.event.kind === 'wrong' ? colors.red : colors.green, 0.9 * (1 - (t - tr) / 3));
      ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
    }
    if (cur.est) {
      const [x, y] = P(S, cur.est[0], cur.est[1]);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = colors.accent; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = colors.surface; ctx.stroke();
    }
    drawRobot(S, sim.gt[t]);

    const badge = $('greedyBadge');
    if (!cur.est) setBadge(badge, 'Lost', '');
    else if (cur.err > 2) setBadge(badge, 'Wrong corridor', 'bad');
    else setBadge(badge, 'Right corridor', 'good');
    $('greedyWrong').textContent = cur.wrongCommits;
    $('greedyWrongLabel').textContent = cur.wrongCommits === 1 ? 'false loop closure' : 'false loop closures';
  }

  // ------------------------------------------------------------ CROSS panel
  function drawCross() {
    const S = C, { ctx, s } = S;
    const f = sim.frames[t];
    const tr = recentRetrieval(t);
    const hl = [];
    const fr = tr >= 0 ? sim.frames[tr] : null;
    const fade = tr >= 0 ? 1 - (t - tr) / 3 : 0;
    if (fr && fr.modes) fr.modes.forEach((m) => m.kfs.forEach((i) => hl.push({ i, color: colors.green, a: fade })));
    drawMap(S, hl);

    // trails of every branch alive up to t
    const committedId = f.committed ? f.committed.id : null;
    const alive = new Map(f.hyps.map((h) => [h.id, h]));
    Object.keys(sim.history).forEach((idStr) => {
      const id = +idStr;
      const pts = sim.history[id];
      if (!pts.length || pts[0].t > t) return;
      const h = alive.get(id);
      const isC = id === committedId;
      const w = h ? h.w : 0;
      ctx.strokeStyle = isC ? colors.accent : alpha(colors.blue, h ? 0.25 + 0.6 * w : 0.18);
      ctx.lineWidth = isC ? 3 : 2;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      let n = 0;
      for (const p of pts) {
        if (p.t > t) break;
        const [x, y] = P(S, p.x, p.y);
        n++ ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      if (n > 1) ctx.stroke();
    });
    // pruned branches: red x where they died
    sim.events.forEach((e) => {
      if (e.type !== 'prune' || e.t > t || t - e.t > 24) return;
      const pts = sim.history[e.id];
      if (!pts || !pts.length) return;
      const last = pts.filter((p) => p.t <= e.t).pop() || pts[0];
      cross(S, last.x, last.y, colors.red, 1 - (t - e.t) / 24);
    });
    // measurement modes (lifted retrievals)
    if (fr && fr.modes) {
      fr.modes.forEach((m) => {
        m.kfs.forEach((i) => {
          const n = sim.map.nodes[i];
          const [nx, ny] = P(S, n.x, n.y), [mx, my] = P(S, m.z[0], m.z[1]);
          ctx.setLineDash([2, 3]); ctx.lineWidth = 1.1; ctx.strokeStyle = alpha(colors.green, 0.8 * fade);
          ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(mx, my); ctx.stroke(); ctx.setLineDash([]);
        });
        ellipse(S, m.z, [0.0569, 0, 0.0569], colors.green, 0.2 * fade, 0.9 * fade, 1.3);
      });
    }
    // live branches
    f.hyps.slice().sort((a, b) => a.w - b.w).forEach((h) => {
      const isC = h.id === committedId;
      const col = isC ? colors.accent : colors.blue;
      ellipse(S, h.mu, h.P, col, 0.08 + 0.3 * h.w, 0.45 + 0.55 * h.w, isC ? 2 : 1.5);
      const [x, y] = P(S, h.mu[0], h.mu[1]);
      ctx.font = `600 ${Math.max(10, s * 0.46)}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillStyle = alpha(colors.ink, 0.55 + 0.45 * h.w);
      ctx.fillText(`#${h.id}`, x + 8, y - 6);
    });
    // commit marker
    if (f.committed) {
      const pts = sim.history[f.committed.id];
      const at = pts.find((p) => p.t === f.committed.t);
      if (at) {
        const [x, y] = P(S, at.x, at.y);
        ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = colors.good; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x - 3.5, y); ctx.lineTo(x - 1, y + 2.8); ctx.lineTo(x + 3.8, y - 2.8); ctx.stroke();
      }
    }
    drawRobot(S, sim.gt[t]);

    const badge = $('crossBadge');
    if (f.committed) setBadge(badge, 'Relocalized', 'good');
    else if (f.hyps.length > 1) setBadge(badge, `Waiting · ${f.hyps.length} branches`, 'wait');
    else if (f.hyps.length === 1) setBadge(badge, 'Waiting · 1 branch', 'wait');
    else setBadge(badge, 'No retrieval yet', '');
    renderHypList(f);
  }

  function setBadge(node, label, cls) {
    if (node.textContent !== label) node.textContent = label;
    const c = 'badge' + (cls ? ' ' + cls : '');
    if (node.className !== c) node.className = c;
  }

  // ------------------------------------------------------------ branch list
  const list = $('hypList');
  function renderHypList(f) {
    const { W, R_NEED } = sim.params;
    list.replaceChildren();
    const hyps = f.hyps.slice().sort((a, b) => a.id - b.id).slice(0, 4);
    if (!hyps.length) {
      const d = document.createElement('div'); d.className = 'hyp-empty';
      d.textContent = 'No branches yet: waiting for the first retrieval.';
      list.appendChild(d);
    }
    hyps.forEach((h) => {
      const isC = f.committed && f.committed.id === h.id;
      const row = document.createElement('div');
      row.className = 'hyp' + (isC ? ' committed' : '');
      const name = document.createElement('span'); name.className = 'hyp-name';
      name.appendChild(document.createElement('i'));
      name.appendChild(document.createTextNode(`Branch #${h.id}`));
      const bar = document.createElement('span'); bar.className = 'hyp-bar';
      const b = document.createElement('b'); b.style.width = (h.w * 100).toFixed(1) + '%'; bar.appendChild(b);
      const wv = document.createElement('span'); wv.className = 'hyp-w'; wv.textContent = Math.round(h.w * 100) + '%';
      const win = document.createElement('span'); win.className = 'hyp-win';
      const sup = h.support.slice(-W);
      for (let k = 0; k < W; k++) {
        const cell = document.createElement('i');
        const idx = k - (W - sup.length);
        if (idx >= 0 && sup[idx]) cell.className = 'on';
        win.appendChild(cell);
      }
      const em = document.createElement('em');
      em.textContent = isC ? 'committed' : `${h.count}/${W} · need ${R_NEED}`;
      win.appendChild(em);
      row.append(name, bar, wv, win);
      list.appendChild(row);
    });
    // latest event
    const ev = sim.events.filter((e) => e.t <= t && t - e.t < 30 && e.t > 2 && e.type !== 'merge' && e.type !== 'birth').pop();
    if (ev) {
      const d = document.createElement('div'); d.className = 'hyp-empty';
      if (ev.type === 'commit') d.textContent = `Branch #${ev.id} held the belief for ${R_NEED} of ${W} updates and was promoted.`;
      else if (ev.type === 'prune' && ev.kind === 'alias') d.textContent = `Branch #${ev.id} pruned: odometry turned where its corridor has no turn.`;
      else if (ev.type === 'prune') d.textContent = `Branch #${ev.id} pruned: a one-off retrieval that nothing confirmed.`;
      list.appendChild(d);
    }
  }

  // ------------------------------------------------------------ error chart
  const errBox = $('errChart');
  let errSvg = null, errGeom = null, playhead = null;
  function buildErr() {
    const { el, text } = window.CrossCharts;
    errBox.replaceChildren();
    const W = Math.max(280, errBox.clientWidth);
    const svg = el('svg', { role: 'img', 'aria-label': 'Position error over time for both methods' }, errBox);
    const lh = window.CrossCharts.legend(svg, [
      { kind: 'line', stroke: 'var(--bar-muted-2)', label: 'Commit on retrieval' },
      { kind: 'line', stroke: 'var(--accent)', label: 'CROSS (after commit)' },
    ], 34, 9, W - 40);
    const left = 34, right = 10, top = lh + 16, bottom = 22, H = top + 102 + bottom;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const N = sim.N, maxE = 6.5;
    const x = (k) => left + (W - left - right) * k / (N - 1);
    const y = (e) => top + (H - top - bottom) * (1 - Math.min(e, maxE) / maxE);
    errGeom = { x, y, W, H, left, right, top, bottom };
    [0, 2, 4, 6].forEach((v) => {
      el('line', { x1: left, x2: W - right, y1: y(v), y2: y(v), class: v === 0 ? 'c-axis' : 'c-grid' }, svg);
      text(svg, left - 6, y(v), v + ' m', 'c-tick', 'end');
    });
    // pre-commit band
    const first = sim.retr.findIndex((r) => r && r.length);
    const ct = sim.summary.commitT;
    if (ct) {
      el('rect', { x: x(first), y: top, width: x(ct) - x(first), height: H - top - bottom, fill: 'var(--blue)', opacity: 0.07 }, svg);
      if (x(ct) - x(first) > 280) text(svg, (x(first) + x(ct)) / 2, y(3.2), 'CROSS holds its branches, commits nothing', 'c-tick c-halo', 'middle');
      text(svg, x(ct) + 4, y(6.1), 'commit', 'c-tick');
      el('line', { x1: x(ct), x2: x(ct), y1: top, y2: H - bottom, class: 'c-axis' }, svg);
    }
    // turn marker
    const turn = sim.gt.findIndex((p) => p[0] > 17.2);
    text(svg, x(turn), H - 8, 'turn ↓', 'c-tick', 'middle');
    // greedy line
    let d = '';
    sim.greedy.forEach((g, k) => { if (g.err == null) return; d += (d ? 'L' : 'M') + x(k).toFixed(1) + ',' + y(g.err).toFixed(1); });
    el('path', { d, fill: 'none', stroke: 'var(--bar-muted-2)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
    let dc = '';
    sim.frames.forEach((f, k) => { if (f.err == null) return; dc += (dc ? 'L' : 'M') + x(k).toFixed(1) + ',' + y(f.err).toFixed(1); });
    if (dc) el('path', { d: dc, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, svg);
    playhead = el('line', { y1: top - 4, y2: H - bottom, stroke: 'var(--ink)', 'stroke-width': 1.25 }, svg);
    // hover readout
    const hit = el('rect', { x: left, y: top, width: W - left - right, height: H - top - bottom, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (e) => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) * (W / r.width);
      const k = Math.max(0, Math.min(N - 1, Math.round((px - left) / (W - left - right) * (N - 1))));
      const g = sim.greedy[k].err, c = sim.frames[k].err;
      window.CrossCharts.tip.show(e, [
        ['tt-v', g == null ? 'no estimate yet' : g.toFixed(1) + ' m'], ['tt-n', 'Commit on retrieval'],
        ['tt-v', c == null ? 'no commitment yet' : c.toFixed(2) + ' m'], ['tt-n', 'CROSS'],
        ['tt-s', `step ${k}`],
      ]);
    });
    hit.addEventListener('pointerleave', () => window.CrossCharts.tip.hide());
    errSvg = svg;
  }

  // ------------------------------------------------------------ loop
  function render() {
    drawGreedy();
    drawCross();
    if (playhead) { const x = errGeom.x(t); playhead.setAttribute('x1', x); playhead.setAttribute('x2', x); }
    scrub.value = t;
    scrub.style.setProperty('--pct', (100 * t / (sim.N - 1)) + '%');
  }
  function frame(ts) {
    if (!playing) return;
    const dt = Math.min(0.1, (ts - (lastTs || ts)) / 1000);
    lastTs = ts;
    acc += dt * STEPS_PER_SEC * speed;
    let changed = false;
    while (acc >= 1) { acc -= 1; if (t < sim.N - 1) { t++; changed = true; } }
    if (changed) render();
    if (t >= sim.N - 1) { setPlaying(false); return; }
    requestAnimationFrame(frame);
  }
  function setPlaying(p) {
    if (p && t >= sim.N - 1) t = 0;
    playing = p;
    playBtn.classList.toggle('playing', p);
    playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
    if (p) { lastTs = 0; acc = 0; requestAnimationFrame(frame); }
  }
  function load(seed) {
    sim = window.CrossSim.simulate(seed);
    scrub.max = sim.N - 1;
    t = 0;
    buildErr();
    render();
  }
  function resize() {
    G = setup(cvG); C = setup(cvC);
    buildErr();
    render();
  }

  playBtn.addEventListener('click', () => setPlaying(!playing));
  $('demoRestart').addEventListener('click', () => { t = 0; render(); setPlaying(true); });
  $('demoReseed').addEventListener('click', () => { load(1 + Math.floor(Math.random() * 300)); setPlaying(true); });
  scrub.addEventListener('input', () => { setPlaying(false); t = +scrub.value; render(); });
  root.querySelectorAll('[data-speed]').forEach((b) => b.addEventListener('click', () => {
    speed = +b.dataset.speed;
    root.querySelectorAll('[data-speed]').forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-checked', o === b); });
  }));

  readColors();
  G = setup(cvG); C = setup(cvC);
  load(7);

  let rT = null;
  window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(resize, 120); });
  document.addEventListener('themechange', () => { readColors(); render(); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); render(); });

  // autoplay once when scrolled into view; pause when it leaves
  let autoplayed = false;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((ents) => {
      ents.forEach((e) => {
        if (e.isIntersecting && e.intersectionRatio > 0.35 && !autoplayed && !reduceMotion) { autoplayed = true; setPlaying(true); }
        else if (!e.isIntersecting && playing) setPlaying(false);
      });
    }, { threshold: [0, 0.35] }).observe(cvC);
  }
})();
