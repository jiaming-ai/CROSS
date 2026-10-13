/* Closed-loop navigation episode viewer (cross_nav.policy.sim.viewer). Vanilla JS + canvas, works from file://.
 *
 * Frames. World plane (x, z), drawn with x to the right and z downward (a true top view of Habitat's Y-up world).
 * Robot frame (forward, left): forward = (-sin yaw, -cos yaw), left = (-cos yaw, sin yaw) in (x, z).
 * BEV debug maps: n x n cells of res metres, row 0 = far forward, column 0 = far left, the robot at the map centre;
 * cell (i, j) centre: forward = (n/2 - i - 0.5) res, left = (n/2 - j - 0.5) res.
 * Topdown bitmap: element (r, c) is the navigability of the point (origin_x + c mpp, origin_z + r mpp).
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------------------------------ pure geometry
  function axes(yaw) {
    const s = Math.sin(yaw), c = Math.cos(yaw);
    return { F: [-s, -c], L: [-c, s] };
  }

  function toWorld(p, pos, yaw) {
    const { F, L } = axes(yaw);
    return [pos[0] + p[0] * F[0] + p[1] * L[0], pos[1] + p[0] * F[1] + p[1] * L[1]];
  }

  /* BEV image coordinates (u = column, v = row, cell corners at integers) -> world:
   * x = ax u + cx v + ex,  z = az u + cz v + ez  (world = pos + (n/2 - v) res F + (n/2 - u) res L). */
  function bevAffine(pos, yaw, n, res) {
    const { F, L } = axes(yaw), h = n / 2 * res;
    return { ax: -res * L[0], az: -res * L[1], cx: -res * F[0], cz: -res * F[1],
      ex: pos[0] + h * (F[0] + L[0]), ez: pos[1] + h * (F[1] + L[1]) };
  }

  /* Robot-frame point (forward, left, up above the floor) -> camera coordinates (x right, y up, z backward; OpenGL),
   * camera at `height` above the robot, rotated by pitch about x (up positive), then roll about z (hab.render). */
  function camPoint(f, l, up, cam) {
    const x = -l, y = up - cam.height, z = -f;
    const cp = Math.cos(cam.pitch || 0), sp = Math.sin(cam.pitch || 0);
    const y1 = cp * y + sp * z, z1 = -sp * y + cp * z;
    const cr = Math.cos(cam.roll || 0), sr = Math.sin(cam.roll || 0);
    return [cr * x + sr * y1, -sr * x + cr * y1, z1];
  }

  /* Floor polyline (forward, left) -> image polylines [[u, v], ...] for a W x H pinhole with horizontal FoV hfov
   * (degrees, square pixels), clipped at the near plane. */
  function projectFloor(pts, cam, W, H, near) {
    near = near || 0.05;
    const foc = (W / 2) / Math.tan((cam.hfov || 90) * Math.PI / 360);
    const P = q => [W / 2 + foc * q[0] / -q[2], H / 2 - foc * q[1] / -q[2]];
    const cams = pts.map(p => (p && isFinite(p[0]) && isFinite(p[1])) ? camPoint(p[0], p[1], 0, cam) : null);
    const out = [];
    let cur = [];
    for (let i = 0; i + 1 < cams.length; i++) {
      const a = cams[i], b = cams[i + 1];
      if (!a || !b) { if (cur.length) out.push(cur); cur = []; continue; }
      const da = -a[2], db = -b[2];
      if (da < near && db < near) { if (cur.length) out.push(cur); cur = []; continue; }
      let A = a, B = b;
      if (da < near) { const t = (near - da) / (db - da); A = a.map((v, k) => v + t * (b[k] - v)); }
      if (db < near) { const t = (near - da) / (db - da); B = a.map((v, k) => v + t * (b[k] - v)); }
      if (!cur.length || da < near) { if (cur.length) out.push(cur); cur = [P(A)]; }
      cur.push(P(B));
      if (db < near) { out.push(cur); cur = []; }
    }
    if (cur.length) out.push(cur);
    return out;
  }

  const pure = { axes, toWorld, bevAffine, camPoint, projectFloor };
  if (typeof document === 'undefined') {           // node: unit tests of the geometry
    if (typeof module !== 'undefined') module.exports = pure;
    return;
  }

  // ------------------------------------------------------------------------------------------ state
  const V = window.VIEWER = { index: null, eps: {}, pending: {}, pure };
  const S = { key: null, step: 0, maxT: 1, overlay: 'occ', playing: false, timer: null, speed: 5, scope: 'all',
    follow: false, theme: 'auto' };
  const $ = id => document.getElementById(id);
  const imgCache = new Map();
  const STATUS = ['route to goal', 'local waypoint', 'no route: rotate'];

  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  }
  const pct = x => (x == null || !isFinite(x)) ? '–' : (100 * x).toFixed(1) + '%';
  const num = (x, d) => (x == null || !isFinite(x)) ? '–' : Number(x).toFixed(d == null ? 2 : d);
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function rgba(name) {
    const s = css(name);
    if (s[0] === '#') {
      const v = s.length === 4 ? s.slice(1).split('').map(c => c + c).join('') : s.slice(1, 7);
      return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16), 255];
    }
    const m = s.match(/[\d.]+/g) || [0, 0, 0, 1];
    return [+m[0], +m[1], +m[2], Math.round(255 * (m[3] == null ? 1 : +m[3]))];
  }

  // ------------------------------------------------------------------------------------------ data loading
  function b64bytes(s) {
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  async function inflate(s) {
    const ds = new DecompressionStream('deflate');
    const buf = await new Response(new Blob([b64bytes(s)]).stream().pipeThrough(ds)).arrayBuffer();
    return new Uint8Array(buf);
  }
  async function prepare(ep) {
    if (ep.topdown) {
      const [H, W] = ep.topdown.shape, bits = b64bytes(ep.topdown.bits), rb = Math.ceil(W / 8);
      const nav = new Uint8Array(H * W);
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) nav[r * W + c] = (bits[r * rb + (c >> 3)] >> (7 - (c & 7))) & 1;
      ep.topdown.nav = nav;
    }
    for (const run of Object.values(ep.runs)) {
      if (!run.bev) continue;
      run.bev.data = {};
      if (typeof DecompressionStream === 'undefined') { run.bev.unsupported = true; continue; }
      for (const [k, m] of Object.entries(run.bev.maps)) {
        const q = await inflate(m.data), L = m.levels || 255, f = new Float32Array(q.length);
        for (let i = 0; i < q.length; i++) f[i] = q[i] / L;
        run.bev.data[k] = f;
      }
    }
  }
  V.addEpisode = function (ep) {
    prepare(ep).then(() => {
      V.eps[ep.key] = ep;
      const p = V.pending[ep.key];
      if (p) p.resolve(ep);
    }, err => { const p = V.pending[ep.key]; if (p) p.reject(err); });
  };
  function loadEpisode(key) {
    if (V.eps[key]) return Promise.resolve(V.eps[key]);
    if (V.pending[key]) return V.pending[key].promise;
    const p = {};
    p.promise = new Promise((res, rej) => { p.resolve = res; p.reject = rej; });
    V.pending[key] = p;
    const s = document.createElement('script');
    s.src = 'data/' + key + '.js';
    s.onerror = () => p.reject(new Error('cannot load ' + s.src));
    document.head.appendChild(s);
    return p.promise;
  }
  V.setIndex = function (idx) {
    V.index = idx;
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  };

  // ------------------------------------------------------------------------------------------ summary
  function renderSummary() {
    const idx = V.index, sum = S.scope === 'common' && idx.common ? idx.common : idx.summary;
    const runs = idx.runs.map(r => r.name);
    const cols = [
      ['Episodes', a => a.n, v => v, null],
      ['Success', a => a.success, pct, 1],
      ['SPL', a => a.spl, v => num(v, 3), 1],
      ['Collision-free success', a => a.collision_free, pct, 1],
      ['Collisions / ep', a => a.collisions, v => num(v, 1), -1],
      ['Steps / ep', a => a.steps, v => num(v, 0), null],
      ['Final geodesic (median, m)', a => a.final_geo_median, v => num(v, 2), -1],
    ];
    const best = (vals, dir) => {
      const f = vals.filter(v => v != null && isFinite(v));
      if (!dir || runs.length < 2 || f.length < 2) return null;
      const lo = Math.min(...f), hi = Math.max(...f);
      if (lo === hi) return null;                    // a tie everywhere highlights nothing
      return dir > 0 ? hi : lo;
    };
    const t = $('sum-table');
    t.replaceChildren(h('thead', null, h('tr', null, h('th', { text: 'Run' }), cols.map(c => h('th', { text: c[0] })))));
    const tb = h('tbody');
    const allVals = cols.map(c => runs.map(r => sum[r].all.n ? c[1](sum[r].all) : null));
    const bests = cols.map((c, i) => best(allVals[i], c[3]));
    runs.forEach((r, ri) => {
      const a = sum[r].all;
      tb.append(h('tr', null, h('td', { class: 'run', text: r }), cols.map((c, i) => {
        const v = a.n ? c[1](a) : null;
        return h('td', { class: (v != null && v === bests[i]) ? 'best' : null, text: a.n || i === 0 ? c[2](v) : '–' });
      })));
    });
    t.append(tb);

    const bt = $('bin-table');
    const bcols = cols.slice(0, 5);
    bt.replaceChildren(h('thead', null, h('tr', null, h('th', { text: 'Geodesic' }), h('th', { class: 'left', text: 'Run' }),
      bcols.map(c => h('th', { text: c[0] })))));
    const bb = h('tbody');
    const nb = sum[runs[0]].bins.length;
    for (let b = 0; b < nb; b++) {
      const bin = sum[runs[0]].bins[b];
      const label = bin.hi == null ? `≥ ${bin.lo} m` : `${bin.lo}–${bin.hi} m`;
      const vals = bcols.map(c => runs.map(r => sum[r].bins[b].n ? c[1](sum[r].bins[b]) : null));
      const bests2 = bcols.map((c, i) => best(vals[i], c[3]));
      runs.forEach((r, ri) => {
        const a = sum[r].bins[b];
        bb.append(h('tr', { class: ri === 0 ? 'group-start' : null },
          h('td', { class: 'bin', text: ri === 0 ? label : '' }), h('td', { class: 'run', text: r }),
          bcols.map((c, i) => {
            const v = a.n ? c[1](a) : null;
            return h('td', { class: (v != null && v === bests2[i]) ? 'best' : (a.n ? null : 'muted'), text: i === 0 ? String(a.n) : (a.n ? c[2](v) : '–') });
          })));
      });
    }
    bt.append(bb);
    const note = [];
    if (idx.common) note.push(S.scope === 'common' ? `Restricted to the ${idx.n_common} episodes every run finished.`
      : `Runs finished different episode sets; "Common" restricts to the ${idx.n_common} shared episodes.`);
    note.push('Success: within 0.2 m of the goal. SPL: success × geodesic / max(geodesic, travelled). Collision-free: success with no blocked step.');
    $('summary-note').textContent = note.join(' ');
  }

  function renderScope() {
    const seg = $('scope-seg');
    if (!V.index.common) { seg.hidden = true; return; }
    seg.hidden = false;
    seg.replaceChildren(...[['all', 'All episodes'], ['common', `Common (${V.index.n_common})`]].map(([k, t]) =>
      h('button', { type: 'button', 'aria-pressed': String(S.scope === k), text: t,
        onclick: () => { S.scope = k; renderScope(); renderSummary(); } })));
  }

  // ------------------------------------------------------------------------------------------ episode list
  function renderList() {
    const ul = $('ep-list');
    ul.replaceChildren();
    for (const e of V.index.episodes) {
      const badges = V.index.runs.map(r => {
        const x = e.runs[r.name];
        if (!x) return h('span', { class: 'badge', title: `${r.name}: not run` }, h('span', { class: 'nm', text: r.name }), h('b', { text: '–' }));
        const t = `${r.name}: ${x.success ? 'success' : 'failure'}, ${x.steps} steps, ${x.collisions} collisions, SPL ${num(x.spl, 2)}${x.viz ? '' : ' (no frames)'}`;
        return h('span', { class: 'badge ' + (x.success ? 'ok' : 'fail'), title: t },
          h('span', { class: 'nm', text: r.name }), h('b', { text: x.success ? '✓' : '✗' }));
      });
      ul.append(h('li', null, h('button', { type: 'button', role: 'option', 'aria-selected': String(e.key === S.key),
        'data-key': e.key, onclick: () => selectEpisode(e.key, 0) },
      h('span', { class: 'ep-id' }, h('span', { text: e.id }), h('span', { class: 'sub', text: num(e.geodesic, 1) + ' m' })),
      h('span', { class: 'badges' }, badges))));
    }
    const n = V.index.episodes.length, tot = Math.max(...V.index.runs.map(r => r.n_viz), 0);
    $('ep-note').textContent = `${n} episode${n === 1 ? '' : 's'} with recorded frames` + (tot > n ? ` (of ${tot}; --max-episodes)` : '') + '.';
  }

  // ------------------------------------------------------------------------------------------ episode view
  let panels = [];

  async function selectEpisode(key, step) {
    stop();
    S.key = key;
    for (const b of document.querySelectorAll('#ep-list button')) b.setAttribute('aria-selected', String(b.dataset.key === key));
    const row = V.index.episodes.find(e => e.key === key);
    $('ep-title').textContent = row ? row.id : key;
    $('ep-meta').textContent = 'loading…';
    $('panels').replaceChildren(h('div', { class: 'card missing', text: 'Loading episode…' }));
    let ep;
    try { ep = await loadEpisode(key); } catch (err) {
      $('panels').replaceChildren(h('div', { class: 'card missing', text: String(err) }));
      return;
    }
    if (S.key !== key) return;
    $('ep-meta').textContent = `scene ${ep.scene} · geodesic ${num(ep.geodesic, 2)} m`;
    S.maxT = Math.max(...Object.values(ep.runs).map(r => r.T));
    const sl = $('step');
    sl.max = String(S.maxT - 1);
    buildPanels(ep);
    setStep(Math.min(step || 0, S.maxT - 1));
  }

  function buildPanels(ep) {
    const box = $('panels');
    box.replaceChildren();
    panels = [];
    for (const r of V.index.runs) {
      const run = ep.runs[r.name];
      const res = V.index.episodes.find(e => e.key === ep.key).runs[r.name];
      const card = h('article', { class: 'card panel' });
      const head = h('div', { class: 'panel-head' }, h('h3', { text: r.name }));
      if (res) head.append(h('span', { class: 'badge ' + (res.success ? 'ok' : 'fail'), text: res.success ? 'success' : 'failure' }));
      card.append(head);
      if (!run) {
        card.append(h('div', { class: 'missing', text: res ? 'Finished, but no frames were recorded for this run.' : 'This run has no result for this episode.' }));
        box.append(card);
        continue;
      }
      const R = run.result;
      card.append(h('div', { class: 'metrics', text: `${R.steps} steps · SPL ${num(R.spl, 3)} · ${R.collisions} collisions · travelled ${num(R.travelled, 1)} m · final geodesic ${num(R.final_geo, 2)} m · mode ${R.mode}` }));
      const P = { run, name: r.name, ep, card };
      P.top = h('canvas', { class: 'topdown', role: 'img', 'aria-label': `Top-down map, ${r.name}` });
      P.cam = h('canvas', { class: 'camera', role: 'img', 'aria-label': `Camera frame, ${r.name}` });
      P.tl = h('canvas', { class: 'timeline', title: 'Per-step status; click to jump' });
      P.tl.addEventListener('click', ev => {
        const b = P.tl.getBoundingClientRect();
        setStep(Math.floor((ev.clientX - b.left) / b.width * S.maxT));
      });
      P.info = h('dl', { class: 'info' });
      P.thumbs = h('div', { class: 'thumbs' });
      card.append(P.top, P.tl, P.cam, P.info, h('div', null, h('h3', { text: 'Causal window (frame index)' }), P.thumbs));
      box.append(card);
      panels.push(P);
    }
    if (window.ResizeObserver && !buildPanels.ro) {
      let lastW = 0;
      buildPanels.ro = new ResizeObserver(es => {
        const w = Math.round(es[0].contentRect.width);
        if (w !== lastW) { lastW = w; requestAnimationFrame(drawAll); }   // outside the observer callback: no RO loop
      });
      buildPanels.ro.observe(box);
    }
  }

  function setStep(s) {
    if (!isFinite(s)) return;
    S.step = Math.max(0, Math.min(S.maxT - 1, s | 0));
    $('step').value = String(S.step);
    $('step-label').textContent = `step ${S.step} / ${S.maxT - 1}`;
    drawAll();
    writeHash();
    preload();
  }

  function drawAll() {
    for (const P of panels) drawPanel(P);
  }

  function drawPanel(P) {
    const t = Math.min(S.step, P.run.T - 1);
    drawTop(P, t);
    drawTimeline(P, t);
    drawCamera(P, t);
    drawInfo(P, t);
  }

  function sizeCanvas(c, aspect) {
    const dpr = window.devicePixelRatio || 1, w = Math.max(50, c.clientWidth || c.parentNode.clientWidth || 300);
    const hgt = Math.round(w * aspect);
    c.style.height = hgt + 'px';
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hgt * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(hgt * dpr); }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: hgt, dpr };
  }

  function viewBox(P, t) {
    let [x0, z0, x1, z1] = P.ep.view;
    if (S.follow) {
      const p = P.run.pos[t], half = 6;
      x0 = p[0] - half; x1 = p[0] + half; z0 = p[1] - half; z1 = p[1] + half;
    }
    // keep the aspect within [0.6, 1.25] (height / width) by padding the short side
    let w = x1 - x0, hh = z1 - z0;
    if (hh / w > 1.25) { const d = (hh / 1.25 - w) / 2; x0 -= d; x1 += d; w = x1 - x0; }
    if (hh / w < 0.6) { const d = (w * 0.6 - hh) / 2; z0 -= d; z1 += d; hh = z1 - z0; }
    return [x0, z0, x1, z1];
  }

  function navBitmap(ep) {
    const key = S.themeKey;
    if (ep._nav && ep._navKey === key) return ep._nav;
    const td = ep.topdown, [H, W] = td.shape;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d'), im = ctx.createImageData(W, H), col = rgba('--map-free');
    for (let i = 0; i < H * W; i++) if (td.nav[i]) { im.data.set(col, i * 4); }
    ctx.putImageData(im, 0, 0);
    ep._nav = c; ep._navKey = key;
    return c;
  }

  function bevBitmap(P, t, kind) {
    const bev = P.run.bev, n = bev.n, src = bev.data[kind];
    if (!src) return null;
    if (!P._bev) { P._bev = document.createElement('canvas'); P._bev.width = n; P._bev.height = n; }
    const ctx = P._bev.getContext('2d'), im = ctx.createImageData(n, n), d = im.data, off = t * n * n;
    const occ = rgba('--bev-occ'), free = rgba('--bev-free'), obs = rgba('--bev-obs'), err = rgba('--bev-err');
    for (let i = 0; i < n * n; i++) {
      const p = src[off + i];
      let c, a;
      if (kind === 'occ') {
        if (p >= 0.5) { c = occ; a = 0.08 + 0.72 * (2 * p - 1); } else { c = free; a = 0.05 + 0.3 * (1 - 2 * p); }   // ~transparent at 0.5
      } else if (kind === 'obs') { c = obs; a = 0.7 * p; } else { c = err; a = 0.8 * p; }
      d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = Math.round(255 * a);
    }
    ctx.putImageData(im, 0, 0);
    return P._bev;
  }

  function drawTop(P, t) {
    const [x0, z0, x1, z1] = viewBox(P, t);
    const { ctx, w, h: hh, dpr } = sizeCanvas(P.top, (z1 - z0) / (x1 - x0));
    const s = w / (x1 - x0);
    const X = p => [(p[0] - x0) * s, (p[1] - z0) * s];
    const run = P.run, ep = P.ep;
    ctx.clearRect(0, 0, w, hh);
    ctx.fillStyle = css('--map-wall'); ctx.fillRect(0, 0, w, hh);
    if (ep.topdown) {
      const td = ep.topdown, bm = navBitmap(ep);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      // pixel (c, r) covers the square centred on its sample point: top-left at origin - mpp / 2
      ctx.setTransform(dpr * s * td.mpp, 0, 0, dpr * s * td.mpp, dpr * s * (td.origin[0] - td.mpp / 2 - x0), dpr * s * (td.origin[1] - td.mpp / 2 - z0));
      ctx.drawImage(bm, 0, 0);
      ctx.restore();
    }
    // grid lines every metre
    ctx.strokeStyle = css('--map-grid'); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let gx = Math.ceil(x0); gx <= x1; gx++) { const u = (gx - x0) * s; ctx.moveTo(u, 0); ctx.lineTo(u, hh); }
    for (let gz = Math.ceil(z0); gz <= z1; gz++) { const v = (gz - z0) * s; ctx.moveTo(0, v); ctx.lineTo(w, v); }
    ctx.stroke();

    const pos = run.pos[t], yaw = run.yaw[t];
    // BEV overlay at the robot pose of this step
    if (run.bev && S.overlay !== 'none' && !run.bev.unsupported) {
      const bm = bevBitmap(P, t, S.overlay);
      if (bm) {
        const n = run.bev.n, A = bevAffine(pos, yaw, n, run.bev.res);
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.setTransform(dpr * s * A.ax, dpr * s * A.az, dpr * s * A.cx, dpr * s * A.cz, dpr * s * (A.ex - x0), dpr * s * (A.ez - z0));
        ctx.drawImage(bm, 0, 0);
        ctx.restore();
        const corner = (u, v) => X([A.ax * u + A.cx * v + A.ex, A.az * u + A.cz * v + A.ez]);
        ctx.save();
        ctx.setLineDash([4, 4]); ctx.strokeStyle = css('--text-muted'); ctx.lineWidth = 1;
        ctx.beginPath();
        [[0, 0], [n, 0], [n, n], [0, n]].forEach(([u, v], i) => { const q = corner(u, v); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
        ctx.closePath(); ctx.stroke();
        ctx.restore();
      }
    }
    const line = (pts, color, width, dash) => {
      const q = pts.filter(p => p && p[0] != null && p[1] != null);
      if (q.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      q.forEach((p, i) => { const c = X(p); i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]); });
      ctx.stroke(); ctx.restore();
    };
    const dot = (p, r, fill, stroke) => {
      const c = X(p);
      ctx.beginPath(); ctx.arc(c[0], c[1], r, 0, 2 * Math.PI);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
    };
    // full trajectory (faint), executed part (solid)
    const full = run.pos.concat([run.final]);
    line(full, css('--traj-full'), 3);
    const ended = S.step >= run.T - 1;
    line(run.pos.slice(0, t + 1).concat(ended ? [run.final] : []), css('--traj'), 2.5);
    // expert path (dashed) and followed path at this step, from the robot
    const ex = run.expert[t];
    if (ex && ex[0] && ex[0][0] != null) line([pos].concat(ex.map(p => p[0] == null ? null : toWorld(p, pos, yaw))), css('--expert'), 1.6, [5, 4]);
    const pth = [pos].concat(run.path[t].map(p => toWorld(p, pos, yaw)));
    line(pth, css('--path'), 2.5);
    for (const p of pth.slice(1)) dot(p, 2, css('--path'));
    // BEV route end and the goal the planner used (when clipped to the map)
    if (run.bev) {
      const tg = run.bev.target && run.bev.target[t];
      if (tg && tg[0] != null) {
        const c = X(toWorld(tg, pos, yaw));
        ctx.save(); ctx.strokeStyle = css('--target'); ctx.lineWidth = 2; ctx.strokeRect(c[0] - 5, c[1] - 5, 10, 10); ctx.restore();
      }
      const gu = run.bev.goal_used && run.bev.goal_used[t], gr = run.goal_rel[t];
      if (gu && gr && Math.hypot(gu[0] - gr[0], gu[1] - gr[1]) > 0.05) {
        const c = X(toWorld(gu, pos, yaw));
        ctx.save(); ctx.strokeStyle = css('--goal'); ctx.setLineDash([2, 2]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(c[0], c[1], 6, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
      }
    }
    // start, goal (success radius), end, robot
    dot(ep.start, 6, css('--start'), css('--surface'));
    const g = X(ep.goal);
    ctx.save(); ctx.fillStyle = css('--goal'); ctx.globalAlpha = 0.18;
    ctx.beginPath(); ctx.arc(g[0], g[1], Math.max(4, 0.2 * s), 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    dot(ep.goal, 6, css('--goal'), css('--surface'));
    if (ended) {
      const c = X(run.final);
      ctx.save(); ctx.fillStyle = css('--traj'); ctx.fillRect(c[0] - 4, c[1] - 4, 8, 8); ctx.restore();
    }
    const rc = X(pos), rr = Math.max(4, 0.18 * s), { F } = axes(yaw);
    ctx.save();
    ctx.fillStyle = css('--robot'); ctx.globalAlpha = 0.35;
    ctx.beginPath(); ctx.arc(rc[0], rc[1], rr, 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = css('--robot'); ctx.lineWidth = 2; ctx.stroke();
    const tip = [rc[0] + F[0] * Math.max(14, 0.5 * s), rc[1] + F[1] * Math.max(14, 0.5 * s)];
    ctx.beginPath(); ctx.moveTo(rc[0], rc[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke();
    const a = Math.atan2(F[1], F[0]);
    ctx.beginPath(); ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - 7 * Math.cos(a - 0.45), tip[1] - 7 * Math.sin(a - 0.45));
    ctx.lineTo(tip[0] - 7 * Math.cos(a + 0.45), tip[1] - 7 * Math.sin(a + 0.45));
    ctx.closePath(); ctx.fillStyle = css('--robot'); ctx.fill();
    ctx.restore();
    // scale bar
    const m = (x1 - x0) > 12 ? 2 : 1, len = m * s;
    ctx.save();
    ctx.fillStyle = css('--surface'); ctx.globalAlpha = 0.85; ctx.fillRect(6, hh - 24, len + 44, 18); ctx.globalAlpha = 1;
    ctx.fillStyle = css('--text'); ctx.fillRect(10, hh - 16, len, 3);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${m} m`, 14 + len, hh - 11);
    ctx.restore();
  }

  function drawTimeline(P, t) {
    const H = 18, c = P.tl, dpr = window.devicePixelRatio || 1, w = Math.max(50, c.clientWidth || 300);
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(H * dpr); }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const run = P.run, bw = w / S.maxT, st = run.bev && run.bev.status, bz = run.bev && run.bev.blind_zone;
    ctx.fillStyle = css('--surface-2'); ctx.fillRect(0, 0, w, H);
    const col = { 0: css('--st0'), 1: css('--st1'), 2: css('--st2'), x: css('--stx'), p: css('--stp') };
    for (let i = 0; i < run.T; i++) {
      const k = run.expert_step[i] ? 'x' : (st ? st[i] : 'p');
      ctx.fillStyle = col[k] || col.p;
      ctx.fillRect(i * bw, 4, Math.max(1, bw + 0.5), H - 4);
      if (bz && bz[i]) { ctx.fillStyle = css('--text'); ctx.fillRect(i * bw, 0, Math.max(1, bw - 0.5), 3); }
    }
    ctx.fillStyle = css('--text');
    ctx.fillRect(Math.min(w - 2, (Math.min(S.step, S.maxT - 1) + 0.5) * bw - 1), 0, 2, H);
  }

  function frameSrc(run, i) {
    if (run.frames.inline) return run.frames.inline[i];
    return `${run.frames.dir}/${String(i).padStart(4, '0')}.jpg`;
  }
  function getImg(src, onload) {
    let im = imgCache.get(src);
    if (!im) {
      im = new Image();
      im.decoding = 'async';
      im.src = src;
      imgCache.set(src, im);
      if (imgCache.size > 1500) imgCache.delete(imgCache.keys().next().value);
    }
    if (!im.complete && onload) im.addEventListener('load', onload, { once: true });
    return im;
  }
  function preload() {
    for (const P of panels) for (let k = 1; k <= 8; k++) {
      const i = Math.min(S.step + k, P.run.T - 1);
      getImg(frameSrc(P.run, i));
    }
  }

  function drawCamera(P, t) {
    const run = P.run, fw = run.frames.w, fh = run.frames.h;
    const { ctx, w, h: hh } = sizeCanvas(P.cam, fh / fw);
    const src = frameSrc(run, t);
    const im = getImg(src, () => { if (Math.min(S.step, P.run.T - 1) === t) drawCamera(P, t); });
    ctx.clearRect(0, 0, w, hh);
    if (im.complete && im.naturalWidth) ctx.drawImage(im, 0, 0, w, hh);
    else { ctx.fillStyle = css('--surface-2'); ctx.fillRect(0, 0, w, hh); }
    const cam = run.camera;
    if (!cam) return;
    const draw = (pts, color, width, dash) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (dash) ctx.setLineDash(dash);
      ctx.shadowColor = 'rgba(0,0,0,.6)'; ctx.shadowBlur = 2;
      for (const seg of projectFloor(pts, cam, w, hh)) {
        ctx.beginPath(); seg.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
      }
      ctx.restore();
    };
    const ex = run.expert[t];
    if (ex && ex[0] && ex[0][0] != null) draw([[0, 0]].concat(ex), '#ffffff', 2, [6, 5]);
    draw([[0, 0]].concat(run.path[t]), css('--path'), 3);
    // goal marker when in front of the camera
    const g = run.goal_rel[t];
    const q = camPoint(g[0], g[1], 0, cam);
    if (-q[2] > 0.1) {
      const foc = (w / 2) / Math.tan((cam.hfov || 90) * Math.PI / 360);
      const u = w / 2 + foc * q[0] / -q[2], v = hh / 2 - foc * q[1] / -q[2];
      if (u > -20 && u < w + 20 && v > -20 && v < hh + 20) {
        ctx.save(); ctx.fillStyle = css('--goal'); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(u, v, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); ctx.restore();
      }
    }
  }

  function drawInfo(P, t) {
    const run = P.run, rows = [];
    const ended = S.step > run.T - 1;
    const stepTxt = h('span', null, `${t} / ${run.T - 1}`);
    if (ended) stepTxt.append(' ', h('span', { class: 'ended', text: `ended (${run.result.success ? 'success' : 'failure'})` }));
    rows.push(['step', stepTxt]);
    const g = run.goal_rel[t];
    rows.push(['goal (fwd, left)', `${num(g[0], 2)}, ${num(g[1], 2)} m · ${num(Math.hypot(g[0], g[1]), 2)} m away`]);
    if (run.expert_step[t]) rows.push(['command', 'expert path']);
    const b = run.bev;
    if (b) {
      const st = b.status ? b.status[t] : null;
      rows.push(['BEV status', st == null ? '–' : h('span', { class: 'st s' + st, text: STATUS[st] || String(st) })]);
      if (b.blind_zone) rows.push(['blind zone', b.blind_zone[t] ? 'used (floor ahead assumed observed)' : 'not used']);
      if (b.vis_threshold) rows.push(['visibility threshold', num(b.vis_threshold[t], 3)]);
      if (b.log_scale) rows.push(['metric scale', `${num(Math.exp(b.log_scale[t]), 2)} m/unit (${b.odometry_scale && b.odometry_scale[t] ? 'odometry' : 'scale head'})`]);
      if (b.n_views) rows.push(['distinct views', String(b.n_views[t])]);
      if (b.up_head_deg) rows.push(['gravity head vs used', num(b.up_head_deg[t], 1) + '°']);
      if (b.unsupported) rows.push(['overlay', 'this browser lacks DecompressionStream']);
    }
    rows.push(['window', run.window[t].join(', ')]);
    P.info.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', null, v)]));
    // causal window thumbnails (unique frames, newest first)
    const seen = new Map();
    for (const i of run.window[t]) seen.set(i, (seen.get(i) || 0) + 1);
    P.thumbs.replaceChildren(...[...seen.entries()].map(([i, c]) => h('figure', null,
      h('img', { src: frameSrc(run, i), alt: `frame ${i}`, loading: 'lazy' }),
      h('figcaption', { text: `#${i}${c > 1 ? ' ×' + c : ''}` }))));
  }

  // ------------------------------------------------------------------------------------------ playback, hash, theme
  function play() {
    if (S.playing) { stop(); return; }
    if (S.step >= S.maxT - 1) setStep(0);
    S.playing = true;
    $('play-btn').textContent = 'Pause';
    $('play-btn').setAttribute('aria-label', 'Pause');
    S.timer = setInterval(() => {
      if (S.step >= S.maxT - 1) { stop(); return; }
      setStep(S.step + 1);
    }, 1000 / S.speed);
  }
  function stop() {
    S.playing = false;
    clearInterval(S.timer);
    const b = $('play-btn');
    if (b) { b.textContent = 'Play'; b.setAttribute('aria-label', 'Play'); }
  }
  function writeHash() {
    if (!S.key) return;
    const q = new URLSearchParams({ ep: S.key, step: String(S.step), ov: S.overlay });
    if (S.follow) q.set('follow', '1');
    try { history.replaceState(null, '', '#' + q.toString()); } catch (e) { /* file:// in some browsers */ }
  }
  function readHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    return { ep: q.get('ep'), step: +(q.get('step') || 0), ov: q.get('ov'), follow: q.get('follow') === '1' };
  }
  function applyTheme() {
    const root = document.documentElement;
    if (S.theme === 'auto') delete root.dataset.theme; else root.dataset.theme = S.theme;
    $('theme-btn').textContent = 'Theme: ' + S.theme;
    S.themeKey = S.theme + ':' + (matchMedia('(prefers-color-scheme: dark)').matches ? 'd' : 'l');
    drawAll();
  }

  function init() {
    if (init.done) return;
    init.done = true;
    const idx = V.index;
    $('runs-desc').textContent = idx.runs.map(r => `${r.name}: ${r.n_results} episodes`).join(' · ');
    try { S.theme = localStorage.getItem('navviewer-theme') || 'auto'; } catch (e) { S.theme = 'auto'; }
    applyTheme();
    $('theme-btn').addEventListener('click', () => {
      S.theme = { auto: 'light', light: 'dark', dark: 'auto' }[S.theme];
      try { localStorage.setItem('navviewer-theme', S.theme); } catch (e) { /* storage unavailable */ }
      applyTheme();
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
    renderScope();
    renderSummary();
    renderList();
    $('step').addEventListener('input', e => { stop(); setStep(+e.target.value); });
    $('play-btn').addEventListener('click', play);
    $('prev-btn').addEventListener('click', () => { stop(); setStep(S.step - 1); });
    $('next-btn').addEventListener('click', () => { stop(); setStep(S.step + 1); });
    $('overlay').addEventListener('change', e => { S.overlay = e.target.value; drawAll(); writeHash(); });
    $('speed').addEventListener('change', e => { S.speed = +e.target.value; if (S.playing) { stop(); play(); } });
    $('follow').addEventListener('change', e => { S.follow = e.target.checked; drawAll(); writeHash(); });
    document.addEventListener('keydown', e => {
      if (!S.key || e.target.closest('input[type=text], select, textarea')) return;
      if (e.key === 'ArrowRight') { stop(); setStep(S.step + 1); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { stop(); setStep(S.step - 1); e.preventDefault(); }
      else if (e.key === ' ' && e.target.tagName !== 'BUTTON') { play(); e.preventDefault(); }
    });
    window.addEventListener('resize', drawAll);
    const hs = readHash();
    if (hs.ov && ['occ', 'obs', 'err', 'none'].includes(hs.ov)) { S.overlay = hs.ov; $('overlay').value = hs.ov; }
    if (hs.follow) { S.follow = true; $('follow').checked = true; }
    const e0 = idx.episodes.find(e => e.key === hs.ep || e.id === hs.ep) || idx.episodes[0];
    if (e0) selectEpisode(e0.key, hs.step);
    else $('panels').replaceChildren(h('div', { class: 'card missing', text: 'No episode has recorded frames (rollout --viz-dir --viz-every).' }));
  }
})();
