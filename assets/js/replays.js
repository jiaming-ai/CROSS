// CROSS site: trajectory replays (replays/index.html). The data in assets/data/replays/ is built by tools/build.py from
// the published benchmark: index.json (the catalog), one file per mapping sequence (T1) and one per localization
// scene (T2, all its query sessions). Paths are top-down, in metres; the replay draws them on a canvas as time runs.
// The page's HTML lists the sequences and scenes it offers, with their titles and in their order; this script turns
// those lists into the picker.
(function () {
  'use strict';
  var app = document.getElementById('rp-app');
  if (!app || !window.fetch || !window.URLSearchParams) return;

  var DATA = '../assets/data/replays/';
  var DURATION = 24;                                       // seconds for a whole session at 1x
  var DEFAULT_ON = { cross: true, orbslam3: true, rtabmap: true };
  // head markers and legend swatches: a shape per system, so identity never rests on colour alone
  var SHAPES = { cross: 'circle', orbslam3: 'square', rtabmap: 'triangle', mast3r_slam: 'diamond', vggt_slam: 'triangle-down',
    lingbot_map: 'pentagon', amb3r_slam: 'hexagon' };
  // the shipped CROSS mode (configs/modes/*.yaml) with the inputs and estimator of each benchmark configuration
  var MODES = { cross_rgbd: 'rgbd_wheel', 'cross_rgbd+vio': 'rgbd_basalt', cross_rgbd_vo: 'rgbd_dpvo', cross_rgbd_ff: 'rgbd_ff_wheel',
    'cross_rgbd_ff+vio': 'rgbd_ff_basalt', cross_stereo: 'stereo_wheel', 'cross_stereo+vio': 'stereo_basalt', cross_stereo_vo: 'stereo_dpvo',
    cross_stereo_vgio: 'stereo_imu', cross_mono: 'mono_dpvo', cross_mono_ff_odom: 'mono_wheel', 'cross_mono_ff_odom+vio': 'mono_basalt',
    cross_mono_ff_vio: 'mono_vio', cross_mono_ff_vgio: 'mono_imu' };
  // the Inputs filter, as on the leaderboard: a Basalt stereo VIO stream is an odometry input
  var INPUT_GROUP = { camera: 'camera', odometry: 'odometry', vio: 'odometry', imu: 'imu' };
  var INPUT_RANK = { camera: 0, odometry: 0, imu: 0, vio: 1 };  // within a group, the dataset's own odometry first
  var INPUT_WORD = { camera: 'camera only', odometry: 'with odometry', imu: 'with IMU' };
  var GROUPS = ['camera', 'odometry', 'imu'];
  // configuration names in the leaderboard's words (tools/build.py LB_CONFIG_WORDS)
  var CONFIG_WORDS = { 'external odometry': 'dataset odometry', 'stereo VIO odometry': 'Basalt stereo VIO', DA3: 'Depth Anything 3',
    PnP: 'PnP on depth' };
  var SETUPS = ['rgbd', 'stereo', 'mono'];
  var SETUP_WORD = { rgbd: 'RGB-D', stereo: 'stereo', mono: 'mono' };
  var ROVER = { day: 'Day', summer: 'Summer', autumn: 'Autumn', winter: 'Winter', spring: 'Spring', dusk: 'Dusk', night: 'Night',
    'night-light': 'Night with lights' };
  var DEFAULT_QUERY = { 't2-rover-campus_large': 'campus_large_winter_2024-01-27' };
  var ODOMETRY = {
    rover: 'On ROVER the dataset odometry is simulated, as in the benchmark: ground-truth increments with noise (SNR 10). ' +
      'The Basalt stereo VIO configurations get stereo-inertial odometry from the robot’s own stereo pair and IMU instead.',
    kitti: 'On KITTI the dataset odometry is dead reckoning of the car’s OXTS (GPS/INS) velocities and yaw rate, without the ' +
      'GPS positions, so it is close to the ground truth. The Basalt stereo VIO configurations run on the stereo pair and the OXTS IMU instead.',
    openloris: 'On OpenLORIS the odometry is the robot’s own wheel odometry.'
  };
  // the tags of the Systems list, explained under it
  var TAGS = {
    sim3: { text: 'Sim(3)', note: 'aligned with scale, as the benchmark scores the monocular baselines (no metric input).' },
    failed: { text: 'failed', bad: true, note: 'less than 80% of the session tracked: a tracking failure in the benchmark.' },
    crashed: { text: 'run failed', bad: true, note: 'a crash or timeout: every scored frame counts as not localized.' },
    upper: { text: 'upper bound', note: 'no relocalization step, so every query pose counts.' },
    stream: { text: 'one stream', note: 'cannot save a map, so it runs the map session and the query session as one stream. ' +
      'MASt3R-SLAM, VGGT-SLAM 2.0 and AMB3R-SLAM log their poses after their final optimization, which favours them.' },
    mode: { mode: true, note: 'the shipped CROSS mode for these inputs (see the docs).' }
  };
  var MOVING_STEP = 0.005;   // m per ground-truth sample: below it the robot stands still (approximates the benchmark rule)
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS = 'http://www.w3.org/2000/svg';

  function $(id) { return document.getElementById(id); }
  var el = {
    intro: $('rp-intro'), tabs: [].slice.call(document.querySelectorAll('.rp-tab')), panel: $('rp-panel'),
    picks: [].slice.call(document.querySelectorAll('.rp-pick')), list1: $('rp-list-t1'), list2: $('rp-list-t2'),
    mapName: $('rp-map-name'), chips: $('rp-chips'), setup: $('rp-setup'), inputs: $('rp-inputs'),
    radiusFilter: document.querySelector('.rp-filter[data-only="t2"]'), radius: $('rp-radius'), speed: $('rp-speed'),
    context: $('rp-context'), viewer: $('rp-viewer'), scnTitle: $('rp-scn-title'), scnSub: $('rp-scn-sub'), best: $('rp-best'),
    stage: $('rp-stage'), canvas: $('rp-canvas'), msg: $('rp-msg'), summary: $('rp-summary'), key: $('rp-key'),
    play: $('rp-play'), scrub: $('rp-scrub'), pos: $('rp-pos'), restart: $('rp-restart'), tlKey: $('rp-tl-key'),
    tlRows: $('rp-tl-rows'), metric: $('rp-side-metric'), sys: $('rp-sys'), notes: $('rp-notes'), notrun: $('rp-notrun'),
    table: $('rp-table'), tableEl: $('rp-table-el'), tableWhat: $('rp-table-what'), routeKey: null
  };

  var CAT = null, ORDER = {}, ITEMS = {}, D = null, Q = null, ROWS = [], tlHead = null, loadSeq = 0;
  var S = { mode: 't1', t1: null, t2: null, query: {}, setup: 'stereo', inputs: 'camera', radius: 1, on: {}, cfg: {},
    hover: null, tau: reduceMotion ? 1 : 0, playing: false, speed: 1, auto: false, inView: false };
  Object.keys(DEFAULT_ON).forEach(function (k) { S.on[k] = true; });

  // ---------------------------------------------------------------- helpers
  function h(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function pct(v) { return Math.round(v * 100) + '%'; }
  function metres(v) { return v == null ? '–' : (v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : String(Math.round(v))) + ' m'; }
  function dprNow() { return Math.min(window.devicePixelRatio || 1, 2); }
  function npts(P) { return P.length >> 1; }

  // a path as flat centimetre integers (first point absolute, then steps) -> Float64Array [x0, y0, x1, y1, ...] in metres
  function decodePath(a) {
    if (!a || a.length < 2) return null;
    var out = new Float64Array(a.length), x = 0, y = 0;
    for (var i = 0; i < a.length; i += 2) { x += a[i]; y += a[i + 1]; out[i] = x / 100; out[i + 1] = y / 100; }
    return out;
  }
  // the point at fractional index f of a path
  function at(P, f) {
    var n = npts(P), i, u;
    if (n < 2) return { x: P[0], y: P[1] };
    i = clamp(Math.floor(f), 0, n - 2);
    u = clamp(f - i, 0, 1);
    return { x: P[2 * i] + (P[2 * i + 2] - P[2 * i]) * u, y: P[2 * i + 1] + (P[2 * i + 3] - P[2 * i + 1]) * u };
  }
  function segDist(x, y, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay, l = dx * dx + dy * dy, u = l ? clamp(((x - ax) * dx + (y - ay) * dy) / l, 0, 1) : 0;
    return Math.hypot(x - ax - u * dx, y - ay - u * dy);
  }
  // a session's names: campus_large_winter_2024-01-27 -> Winter (2024-01-27); OpenLORIS corridor1-2 -> session 2
  function session(s) {
    var m = /^campus_large_([a-z-]+)_(\d{4}-\d{2}-\d{2})/.exec(s);
    if (m) { var n = ROVER[m[1]] || m[1]; return { chip: n, date: m[2], label: n + ' (' + m[2] + ')' }; }
    m = /^[a-z]+\d+-(\d+)$/.exec(s);
    if (m) return { chip: 'Session ' + m[1], label: 'session ' + m[1] };
    return { chip: s, label: s };
  }
  function capital(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  // ---------------------------------------------------------------- timing of a mapping run (T1)
  // The published paths carry no timestamps, and CROSS keeps one pose per keyframe rather than per time step, so the
  // estimated poses are matched in order to the ground truth (evenly spaced in time) by dynamic time warping. A run that
  // tracked only part of the session (completeness c) is placed where its poses match when it is accurate enough for
  // that to mean something; otherwise it is assumed to start with the session and to cover the fraction c of it.
  // Per pose: the first (a) and last (b) matched time, as fractions of the session.
  function timing(r, gt) {
    var m = npts(gt), c = clamp(r.completeness || 0, 0, 1), W;
    if (c >= 0.95 || m < 3) W = warp(r.e, gt, m - 1, false);
    else if (r.ate != null && r.ate <= 0.1 * Math.min(D.gtDiag, c * D.gtLen)) W = warp(r.e, gt, m - 1, true);
    else W = warp(r.e, gt, Math.max(1, Math.round(c * (m - 1))), false);
    var den = Math.max(1, m - 1);
    for (var i = 0; i < W.a.length; i++) { W.a[i] /= den; W.b[i] /= den; }
    return { a: W.a, b: W.b, end: W.b[W.b.length - 1] };
  }
  // dynamic time warping of the poses e against the ground-truth samples 0..j1; open: the match may begin and end
  // anywhere in them. Returns the first and last matched sample of each pose.
  function warp(e, gt, j1, open) {
    var n = npts(e), w = j1 + 1, Dm = new Float64Array(n * w), dir = new Uint8Array(n * w), i, k;
    for (i = 0; i < n; i++) {
      var ex = e[2 * i], ey = e[2 * i + 1];
      for (k = 0; k < w; k++) {
        var gx = gt[2 * k] - ex, gy = gt[2 * k + 1] - ey, q = i * w + k, b = 0, d = 0;
        if (i && k) {
          b = Dm[q - w - 1];
          if (Dm[q - w] < b) { b = Dm[q - w]; d = 1; }
          if (Dm[q - 1] < b) { b = Dm[q - 1]; d = 2; }
        } else if (i) { b = Dm[q - w]; d = 1; } else if (k && !open) { b = Dm[q - 1]; d = 2; }
        Dm[q] = b + Math.sqrt(gx * gx + gy * gy);
        dir[q] = d;
      }
    }
    k = w - 1;
    if (open) {                                       // the last pose ends at its best-matching sample
      var best = Infinity;
      for (var kk = 0; kk < w; kk++) if (Dm[(n - 1) * w + kk] < best) { best = Dm[(n - 1) * w + kk]; k = kk; }
    }
    var A = new Float64Array(n), B = new Float64Array(n);
    i = n - 1; A[i] = B[i] = k;
    while (i > 0 || (k > 0 && !open)) {
      var dd = dir[i * w + k];
      if (dd === 2) k--; else { i--; if (dd === 0) k--; B[i] = k; }
      A[i] = k;
    }
    return { a: A, b: B };
  }
  function timingOf(r) { return r._t || (r._t = timing(r, D.gt)); }
  // where a run is at time tau: null before its first pose; i = the last pose reached
  function headOf(r, T, tau) {
    var a = T.a, b = T.b, n = a.length, lo = 0, hi = n - 1, e = r.e;
    if (tau < a[0]) return null;
    while (lo < hi) { var mid = (lo + hi + 1) >> 1; if (a[mid] <= tau) lo = mid; else hi = mid - 1; }
    var x = e[2 * lo], y = e[2 * lo + 1];
    if (lo < n - 1 && tau > b[lo]) {
      var u = clamp((tau - b[lo]) / Math.max(1e-9, a[lo + 1] - b[lo]), 0, 1);
      x += (e[2 * lo + 2] - x) * u; y += (e[2 * lo + 3] - y) * u;
    }
    return { i: lo, x: x, y: y };
  }
  function isLost(r, T, tau) { return r.completeness < 0.95 && T.end < 0.985 && tau > T.end + 0.004; }

  // ---------------------------------------------------------------- localization states (T2)
  // The frames the benchmark scores (PROTOCOL.md, T2): those the map covers (within the larger threshold of the map
  // session's path) at which the robot moves. Rebuilt here from the published, subsampled paths.
  function scored(q) {
    if (q._mask) return q._mask;
    var g = q.gt, m = npts(g), M = D.mapGt, nm = npts(M), r = D.thresholds[1], mask = new Uint8Array(m);
    for (var j = 0; j < m; j++) {
      var x = g[2 * j], y = g[2 * j + 1], cov = nm === 1 && Math.hypot(x - M[0], y - M[1]) < r;
      for (var k = 1; k < nm && !cov; k++) cov = segDist(x, y, M[2 * k - 2], M[2 * k - 1], M[2 * k], M[2 * k + 1]) < r;
      var a = Math.max(0, j - 1), b = Math.min(m - 1, j + 1);
      var moving = Math.hypot(g[2 * b] - g[2 * a], g[2 * b + 1] - g[2 * a + 1]) / Math.max(1, b - a) > MOVING_STEP;
      mask[j] = cov && moving ? 1 : 0;
    }
    return (q._mask = mask);
  }
  // Per ground-truth sample of the query session: st 0 = not scored, 1 = localized within the radius, 2 = wrong (beyond
  // it), 3 = no pose; err in m (-1 = no pose). The run's error curve c spans the scored frames only, evenly spaced.
  function states(r) {
    var ri = S.radius;
    r._st = r._st || [];
    if (r._st[ri]) return r._st[ri];
    var mask = scored(Q), m = mask.length, rad = D.thresholds[ri], c = r.c, nc = c ? c.length : 0, kept = 0, q = 0, j;
    var st = new Uint8Array(m), err = new Float32Array(m).fill(-1), ok = new Uint16Array(m + 1), sc = new Uint16Array(m + 1);
    for (j = 0; j < m; j++) kept += mask[j];
    for (j = 0; j < m; j++) {
      if (mask[j]) {
        var v = nc ? c[kept > 1 ? Math.round(q / (kept - 1) * (nc - 1)) : 0] : -1;
        q++;
        err[j] = v < 0 ? -1 : v / 100;
        st[j] = v < 0 ? 3 : v / 100 < rad ? 1 : 2;
      }
      ok[j + 1] = ok[j] + (st[j] === 1 ? 1 : 0);
      sc[j + 1] = sc[j] + mask[j];
    }
    return (r._st[ri] = { st: st, err: err, ok: ok, sc: sc });
  }
  function finalRecall(r) {
    var v = r.lr && r.lr[S.radius];
    if (v != null) return v;
    var s = states(r), m = s.st.length;
    return s.sc[m] ? s.ok[m] / s.sc[m] : 0;
  }
  function stateText(s, j) {
    var st = s.st[j];
    return st === 0 ? 'not scored' : st === 3 ? 'no pose' : metres(s.err[j]) + (st === 2 ? ', wrong' : '');
  }

  // ---------------------------------------------------------------- data
  var cache = {};
  function load(id) {
    if (!cache[id]) {
      cache[id] = fetch(DATA + id + '.json').then(function (res) {
        if (!res.ok) throw new Error(res.status);
        return res.json();
      }).then(prepare);
      cache[id].catch(function () { delete cache[id]; });
    }
    return cache[id];
  }
  function prepare(d) {
    // a run whose VIO odometry fell back to the dataset's odometry (OpenLORIS has no VIO stream) repeats the
    // odometry run, so the page leaves it out
    var keep = function (r) { prepRun(r); return !r.odom_fallback; };
    if (d.track === 't1') {
      var g = d.gt = decodePath(d.gt), b = grow(g, { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity });
      d.gtDiag = Math.hypot(b.x1 - b.x0, b.y1 - b.y0);
      d.gtLen = 0;
      for (var i = 2; i < g.length; i += 2) d.gtLen += Math.hypot(g[i] - g[i - 2], g[i + 1] - g[i - 1]);
      d.runs = d.runs.filter(keep);
      d.runs.forEach(function (r) { r.e = decodePath(r.e); });
    } else {
      d.mapGt = decodePath(d.map_gt);
      d.queries.forEach(function (q) { q.gt = decodePath(q.gt); q.runs = q.runs.filter(keep); });
    }
    return d;
  }
  function prepRun(r) {
    var s = CAT.systems[r.system] || {}, m = /\(([^)]*)\)\s*$/.exec(s.label || '');
    r.fam = s.family || r.system;
    r.inputs = s.inputs || 'camera';
    r.group = INPUT_GROUP[r.inputs] || 'camera';
    r.cfg = m ? m[1].split(',').map(function (p) { p = p.trim(); return CONFIG_WORDS[p] || p; }).join(' · ') : '';
    r.order = ORDER[r.system] == null ? 999 : ORDER[r.system];
    r.upper = !!s.upper_bound;
    r.stream = s.map_reuse === 'concatenated stream';
  }
  function runsNow() { return !D ? [] : D.track === 't1' ? D.runs : Q.runs; }
  function scene(id) { return CAT.localization.filter(function (x) { return x.id === id; })[0] || null; }
  function curQuery() {
    var it = scene(S.t2), q = S.query[S.t2] || DEFAULT_QUERY[S.t2];
    return it && it.queries.indexOf(q) >= 0 ? q : it ? it.queries[0] : null;
  }
  function queryOf(d, q) { return d.queries.filter(function (x) { return x.query === q; })[0] || d.queries[0]; }

  // ---------------------------------------------------------------- rows: one per system family
  function options(runs, fam) {
    return runs.filter(function (r) { return r.fam === fam && r.setup === S.setup && r.group === S.inputs; })
      .sort(function (a, b) {
        return (MODES[a.system] ? 0 : 1) - (MODES[b.system] ? 0 : 1) || INPUT_RANK[a.inputs] - INPUT_RANK[b.inputs] || a.order - b.order;
      });
  }
  // why a system has no run in this view: what it was run with instead
  function notRun(runs, fam) {
    var mine = runs.filter(function (r) { return r.fam === fam; }), here;
    if (!mine.length) return 'not run here';
    here = mine.filter(function (r) { return r.setup === S.setup; });
    if (here.length) {
      return GROUPS.filter(function (g) { return here.some(function (r) { return r.group === g; }); })
        .map(function (g) { return INPUT_WORD[g]; }).join(' or ');
    }
    return SETUPS.filter(function (s) { return mine.some(function (r) { return r.setup === s; }); })
      .map(function (s) { return SETUP_WORD[s]; }).join(', ') + ' only';
  }
  function buildRows() {
    var runs = runsNow();
    ROWS = Object.keys(CAT.families).map(function (fam) {
      var opts = options(runs, fam), want = S.cfg[fam + '|' + S.setup + '|' + S.inputs];
      var run = opts.filter(function (r) { return r.system === want; })[0] || opts[0] || null;
      return { fam: fam, name: CAT.families[fam], run: run, opts: opts, why: run ? '' : notRun(runs, fam) };
    });
  }
  function shown() { return ROWS.filter(function (r) { return r.run && S.on[r.fam]; }); }
  function focusRow() {
    var vis = shown();
    return vis.filter(function (r) { return r.fam === S.hover; })[0] || vis[0] || null;
  }

  // ---------------------------------------------------------------- colours (tokens; re-read when the theme changes)
  var COL = {}, FONT = 'system-ui, sans-serif';
  function readColours() {
    var cs = getComputedStyle(document.documentElement);
    Object.keys(CAT.families).concat('gt').forEach(function (k) { COL[k] = cs.getPropertyValue('--c-' + k).trim(); });
    ['ink', 'ink-2', 'ink-3', 'bad', 'grid', 'line', 'line-2'].forEach(function (k) { COL[k] = cs.getPropertyValue('--' + k).trim(); });
    COL.pane = getComputedStyle(el.stage).backgroundColor;
    FONT = getComputedStyle(document.body).fontFamily || FONT;
  }

  // ---------------------------------------------------------------- shapes
  var SIDES = { square: [4, Math.PI / 4, 1.25], diamond: [4, 0, 1.3], triangle: [3, -Math.PI / 2, 1.35],
    'triangle-down': [3, Math.PI / 2, 1.35], pentagon: [5, -Math.PI / 2, 1.12], hexagon: [6, 0, 1.1] };
  function shapePoints(shape, cx, cy, r) {
    var s = SIDES[shape], pts = [];
    if (!s) return null;                                   // a circle
    for (var i = 0; i < s[0]; i++) {
      var a = s[1] + i * 2 * Math.PI / s[0];
      pts.push([cx + Math.cos(a) * r * s[2], cy + Math.sin(a) * r * s[2]]);
    }
    return pts;
  }
  function swatch(fam) {
    var svg = document.createElementNS(SVGNS, 'svg'), p = document.createElementNS(SVGNS, 'path'), pts = shapePoints(SHAPES[fam], 8, 8, 5);
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('class', 'rp-sw');
    svg.setAttribute('aria-hidden', 'true');
    svg.style.setProperty('--c', 'var(--c-' + fam + ')');
    p.setAttribute('d', pts ? 'M' + pts.map(function (q) { return q[0].toFixed(2) + ' ' + q[1].toFixed(2); }).join('L') + 'Z'
      : 'M3 8a5 5 0 1 0 10 0a5 5 0 1 0-10 0Z');
    svg.appendChild(p);
    return svg;
  }
  function shapePath(c, fam, x, y, r) {
    var pts = shapePoints(SHAPES[fam], x, y, r);
    c.beginPath();
    if (!pts) { c.arc(x, y, r, 0, Math.PI * 2); return; }
    pts.forEach(function (q, i) { if (i) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]); });
    c.closePath();
  }
  // a marker with a 2px ring in the pane colour, so it stays legible where it crosses a line
  function marker(c, fam, x, y) {
    shapePath(c, fam, x, y, 5);
    c.lineJoin = 'round';
    c.lineWidth = 4; c.strokeStyle = COL.pane; c.stroke();
    c.fillStyle = COL[fam]; c.fill();
  }
  function lostMark(c, fam, x, y) {
    c.lineCap = 'round';
    [[5, COL.pane], [2.25, COL[fam]]].forEach(function (s) {
      c.lineWidth = s[0]; c.strokeStyle = s[1]; c.beginPath();
      c.moveTo(x - 4.5, y - 4.5); c.lineTo(x + 4.5, y + 4.5); c.moveTo(x + 4.5, y - 4.5); c.lineTo(x - 4.5, y + 4.5); c.stroke();
    });
  }
  // a run whose position lies outside the view: an arrow at the edge of the view, pointing to it
  function edgeArrow(c, fam, x, y) {
    var cx = V.w / 2, cy = V.h / 2, dx = x - cx, dy = y - cy, m = 12;
    var t = Math.min((cx - m) / Math.max(Math.abs(dx), 1e-6), (cy - m) / Math.max(Math.abs(dy), 1e-6));
    var ax = cx + dx * t, ay = cy + dy * t;
    c.save();
    c.translate(ax, ay); c.rotate(Math.atan2(dy, dx));
    c.beginPath(); c.moveTo(7, 0); c.lineTo(-5, -6); c.lineTo(-2, 0); c.lineTo(-5, 6); c.closePath();
    c.lineJoin = 'round'; c.lineWidth = 4; c.strokeStyle = COL.pane; c.stroke();
    c.fillStyle = COL[fam]; c.fill();
    c.restore();
    return { x: ax, y: ay };
  }

  // ---------------------------------------------------------------- view: equal aspect, fitted to the ground truth
  var V = { w: 0, h: 0, dpr: 1, s: 1, ox: 0, oy: 0, step: 1 };
  var ctx = el.canvas.getContext('2d'), layer = document.createElement('canvas');
  function X(x) { return V.ox + x * V.s; }
  function Y(y) { return V.oy - y * V.s; }
  function grow(P, b) {
    for (var i = 0; i < P.length; i += 2) {
      if (P[i] < b.x0) b.x0 = P[i]; if (P[i] > b.x1) b.x1 = P[i];
      if (P[i + 1] < b.y0) b.y0 = P[i + 1]; if (P[i + 1] > b.y1) b.y1 = P[i + 1];
    }
    return b;
  }
  function niceStep(v) {
    var p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }
  function refit() {
    if (!D || !V.w) return;
    var b = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
    if (D.track === 't1') grow(D.gt, b);
    else {                                             // both routes and the whole shaded area of the map
      grow(Q.gt, b); grow(D.mapGt, b);
      var r = D.thresholds[1];
      b.x0 -= r; b.x1 += r; b.y0 -= r; b.y1 += r;
    }
    var pad = clamp(Math.min(V.w, V.h) * 0.07, 18, 44), bottom = Math.max(pad, 34);
    var w = Math.max(10, V.w - 2 * pad), hh = Math.max(10, V.h - pad - bottom), bw = Math.max(b.x1 - b.x0, 0.5), bh = Math.max(b.y1 - b.y0, 0.5);
    V.s = Math.min(w / bw, hh / bh);
    V.ox = pad + (w - bw * V.s) / 2 - b.x0 * V.s;
    V.oy = pad + (hh + bh * V.s) / 2 + b.y0 * V.s;
    V.step = niceStep(clamp(V.w * 0.14, 70, 130) / V.s);
  }
  function resize() {
    var w = el.stage.clientWidth, hh = el.stage.clientHeight;
    if (!w || !hh) return;
    V.w = w; V.h = hh; V.dpr = dprNow();
    [el.canvas, layer].forEach(function (cv) { cv.width = Math.round(w * V.dpr); cv.height = Math.round(hh * V.dpr); });
    refit(); paintStatic(); requestDraw();
  }
  function trace(c, P, end) {
    var last = end == null ? npts(P) - 1 : end;
    c.beginPath();
    c.moveTo(X(P[0]), Y(P[1]));
    for (var i = 1; i <= last; i++) c.lineTo(X(P[2 * i]), Y(P[2 * i + 1]));
  }
  // the static layer: metric grid, the map session's route (T2), the ground truth, the scale bar
  function paintStatic() {
    var c = layer.getContext('2d');
    c.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
    c.clearRect(0, 0, V.w, V.h);
    if (!D || !V.w) return;
    var st = V.step, x, y;
    c.strokeStyle = COL.grid; c.lineWidth = 1; c.beginPath();
    for (x = Math.ceil(-V.ox / V.s / st) * st; X(x) <= V.w; x += st) { var px = Math.round(X(x)) + 0.5; c.moveTo(px, 0); c.lineTo(px, V.h); }
    for (y = Math.ceil((V.oy - V.h) / V.s / st) * st; Y(y) >= 0; y += st) { var py = Math.round(Y(y)) + 0.5; c.moveTo(0, py); c.lineTo(V.w, py); }
    c.stroke();
    c.lineCap = 'round'; c.lineJoin = 'round';
    if (D.track === 't2') {
      // faintly shaded: where the map can localize the robot (within the larger threshold of the map session's route)
      c.globalAlpha = 0.03; c.strokeStyle = COL.ink; c.lineWidth = 2 * D.thresholds[1] * V.s; trace(c, D.mapGt); c.stroke();
      c.globalAlpha = 0.55; c.strokeStyle = COL['ink-3']; c.lineWidth = 1.5; trace(c, D.mapGt); c.stroke();
      c.globalAlpha = 1;
    }
    c.strokeStyle = COL.gt; c.lineWidth = 1.5; c.setLineDash([5, 4]); c.lineCap = 'butt';
    trace(c, D.track === 't1' ? D.gt : Q.gt); c.stroke();
    c.setLineDash([]);
    // scale bar, bottom left: the grid step
    var L = st * V.s, bx = 16, by = V.h - 16, label = (st >= 1 ? String(st) : String(+st.toPrecision(2))) + ' m';
    c.strokeStyle = COL['ink-2']; c.lineWidth = 1.5; c.lineCap = 'butt'; c.beginPath();
    c.moveTo(bx, by - 4); c.lineTo(bx, by); c.lineTo(bx + L, by); c.lineTo(bx + L, by - 4); c.stroke();
    c.font = '600 11.5px ' + FONT; c.textBaseline = 'middle'; c.lineJoin = 'round';
    c.lineWidth = 3; c.strokeStyle = COL.pane; c.strokeText(label, bx + L + 7, by - 2);
    c.fillStyle = COL['ink-2']; c.fillText(label, bx + L + 7, by - 2);
  }

  // ---------------------------------------------------------------- drawing (one frame)
  function draw() {
    if (!D || !V.w) return;
    ctx.setTransform(V.dpr, 0, 0, V.dpr, 0, 0);
    ctx.clearRect(0, 0, V.w, V.h);
    ctx.drawImage(layer, 0, 0, V.w, V.h);
    if (D.track === 't1') drawMapping(); else drawLocalization();
    drawStrips();
    updateLive();
  }
  function drawMapping() {
    var vis = shown(), hot = vis.some(function (r) { return r.fam === S.hover; }) ? S.hover : null, heads = [];
    // CROSS on top of the baselines, a highlighted system on top of all
    vis.sort(function (a, b) { return (a.fam === hot) - (b.fam === hot) || (a.fam === 'cross') - (b.fam === 'cross'); });
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    vis.forEach(function (row) {
      var r = row.run;
      if (!r.e) return;
      var T = timingOf(r), hd = headOf(r, T, S.tau);
      if (!hd) return;
      ctx.globalAlpha = hot && row.fam !== hot ? 0.16 : 1;
      ctx.strokeStyle = COL[row.fam];
      ctx.lineWidth = row.fam === 'cross' ? 2.75 : 2;
      trace(ctx, r.e, hd.i); ctx.lineTo(X(hd.x), Y(hd.y)); ctx.stroke();
      heads.push({ row: row, x: X(hd.x), y: Y(hd.y), lost: isLost(r, T, S.tau), alpha: ctx.globalAlpha });
    });
    var avoid = [{ x: 0, y: V.h - 34, w: V.step * V.s + 64, h: 34 }];
    heads.forEach(function (hd) {
      var off = hd.x < 0 || hd.x > V.w || hd.y < 0 || hd.y > V.h;
      ctx.globalAlpha = hd.alpha;
      if (off) {
        var p = edgeArrow(ctx, hd.row.fam, hd.x, hd.y);
        hd.x = p.x; hd.y = p.y;
      } else if (hd.lost) lostMark(ctx, hd.row.fam, hd.x, hd.y);
      else marker(ctx, hd.row.fam, hd.x, hd.y);
      hd.text = hd.row.name + (off ? ' · off view' : hd.lost ? ' · lost track' : '');
      avoid.push({ x: hd.x - 7, y: hd.y - 7, w: 14, h: 14 });
    });
    ctx.globalAlpha = 1;
    // where the robot really is now
    var g = at(D.gt, S.tau * (npts(D.gt) - 1)), gx = X(g.x), gy = Y(g.y);
    ctx.beginPath(); ctx.arc(gx, gy, 5.5, 0, Math.PI * 2);
    ctx.lineWidth = 5; ctx.strokeStyle = COL.pane; ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = COL.ink; ctx.stroke();
    avoid.push({ x: gx - 8, y: gy - 8, w: 16, h: 16 });
    heads.sort(function (a, b) { return (b.row.fam === 'cross') - (a.row.fam === 'cross'); });
    placeLabels(heads, avoid);
  }
  function drawLocalization() {
    var g = Q.gt, m = npts(g), f = S.tau * (m - 1), row = focusRow(), p = at(g, f), hx = X(p.x), hy = Y(p.y), s = null;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (row) {
      s = states(row.run);
      var a = 0;
      while (a < f) {                                  // runs of equal state up to the playhead
        var st = s.st[a], b = a + 1;
        while (b < f && s.st[b] === st) b++;
        if (st) {
          var end = Math.min(b, f);
          ctx.globalAlpha = st === 3 ? 0.75 : 1;
          ctx.strokeStyle = st === 1 ? COL[row.fam] : st === 2 ? COL.bad : COL['ink-3'];
          ctx.lineWidth = st === 3 ? 2.25 : 3.5;
          ctx.setLineDash(st === 2 ? [5, 4] : []);
          ctx.beginPath(); ctx.moveTo(X(g[2 * a]), Y(g[2 * a + 1]));
          for (var k = a + 1; k <= Math.floor(end); k++) ctx.lineTo(X(g[2 * k]), Y(g[2 * k + 1]));
          if (end > Math.floor(end)) { var q = at(g, end); ctx.lineTo(X(q.x), Y(q.y)); }
          ctx.stroke();
        }
        a = b;
      }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    // the robot's true position with the success radius around it: a dashed outline, filled only while it is small
    var rpx = D.thresholds[S.radius] * V.s;
    if (rpx > 6) {
      ctx.beginPath(); ctx.arc(hx, hy, rpx, 0, Math.PI * 2);
      if (rpx < 0.15 * Math.min(V.w, V.h)) { ctx.globalAlpha = 0.06; ctx.fillStyle = COL.ink; ctx.fill(); }
      ctx.globalAlpha = 0.7; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.strokeStyle = COL['ink-3']; ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
    ctx.lineWidth = 4; ctx.strokeStyle = COL.pane; ctx.stroke(); ctx.fillStyle = COL.ink; ctx.fill();
    if (row) {
      placeLabels([{ text: row.name + ': ' + stateText(s, Math.round(f)), x: hx, y: hy, alpha: 1 }],
        [{ x: 0, y: V.h - 34, w: V.step * V.s + 64, h: 34 }]);
    }
  }
  // direct labels beside the markers, where they fit (never over another label or marker)
  function placeLabels(items, avoid) {
    var boxes = avoid.slice(), hh = 16;
    ctx.font = '600 12px ' + FONT; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
    items.forEach(function (it) {
      var w = ctx.measureText(it.text).width + 2, cands = [[10, -11], [10, 11], [-10 - w, -11], [-10 - w, 11], [-w / 2, -19], [-w / 2, 19]];
      for (var i = 0; i < cands.length; i++) {
        var b = { x: it.x + cands[i][0], y: it.y + cands[i][1] - hh / 2, w: w, h: hh };
        if (b.x < 4 || b.y < 4 || b.x + b.w > V.w - 4 || b.y + b.h > V.h - 4) continue;
        if (boxes.some(function (o) { return b.x < o.x + o.w && o.x < b.x + b.w && b.y < o.y + o.h && o.y < b.y + b.h; })) continue;
        boxes.push(b);
        ctx.globalAlpha = it.alpha;
        ctx.lineWidth = 3.5; ctx.strokeStyle = COL.pane; ctx.strokeText(it.text, b.x + 1, b.y + hh / 2);
        ctx.fillStyle = COL.ink; ctx.fillText(it.text, b.x + 1, b.y + hh / 2);
        break;
      }
    });
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- timeline strips (one per shown system)
  function buildStrips() {
    var vis = shown();
    el.tlRows.replaceChildren();
    ROWS.forEach(function (r) { r.strip = null; });
    if (!vis.length) el.tlRows.appendChild(h('p', 'rp-tl-empty', 'Turn on a system in the Systems list to see its timeline.'));
    vis.forEach(function (row) {
      var div = h('div', 'rp-tl-row'), lab = h('span', 'rp-tl-label'), cv = h('canvas', 'rp-tl-strip'), val = h('span', 'rp-tl-val');
      lab.append(swatch(row.fam), h('span', null, row.name));
      div.append(lab, cv, val);
      div.dataset.fam = row.fam;
      div.classList.toggle('is-hot', row.fam === S.hover);
      row.strip = { cv: cv, img: document.createElement('canvas'), val: val, text: null, w: 0, h: 0 };
      el.tlRows.appendChild(div);
    });
    tlHead = vis.length ? h('div', 'rp-tl-head') : null;
    if (tlHead) el.tlRows.appendChild(tlHead);
    el.tlRows.classList.toggle('is-dim', vis.some(function (r) { return r.fam === S.hover; }));
    paintStrips();
  }
  function paintStrips() {
    if (!D) return;
    var dpr = dprNow();
    shown().forEach(function (row) {
      var sp = row.strip;
      if (!sp) return;
      sp.w = sp.cv.clientWidth; sp.h = sp.cv.clientHeight;
      if (!sp.w || !sp.h) return;
      [sp.cv, sp.img].forEach(function (cv) { cv.width = Math.round(sp.w * dpr); cv.height = Math.round(sp.h * dpr); });
      var c = sp.img.getContext('2d');
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (D.track === 't1') paintTrack(c, row.run, row.fam, sp); else paintStates(c, row.run, row.fam, sp);
    });
    drawStrips();
  }
  // T1: the stretch of the session a run tracked, an x where it lost track; right: the share of the session tracked
  function paintTrack(c, r, fam, sp) {
    var mid = sp.h / 2;
    c.fillStyle = COL.line; c.fillRect(0, mid - 0.5, sp.w, 1);
    sp.val.textContent = pct(r.completeness || 0);
    if (!r.e) return;
    var T = timingOf(r), x0 = T.a[0] * sp.w, x1 = T.end * sp.w;
    c.fillStyle = COL[fam];
    c.beginPath();
    if (c.roundRect) c.roundRect(x0, mid - 4, Math.max(2, x1 - x0), 8, 4); else c.rect(x0, mid - 4, Math.max(2, x1 - x0), 8);
    c.fill();
    if (isLost(r, T, 1)) lostMark(c, fam, Math.min(x1 + 6, sp.w - 6), mid);
  }
  // T2, a win/loss strip: localized = a bar up in the system's colour, wrong = a bar down in --bad, no pose = a short
  // neutral tick, not scored = nothing (position, not only colour, tells the states apart); right: localized so far
  function paintStates(c, r, fam, sp) {
    var s = states(r), m = s.st.length, mid = Math.round(sp.h / 2), unit = sp.w / Math.max(1, m - 1), j = 0;
    c.fillStyle = COL.line; c.fillRect(0, mid - 0.5, sp.w, 1);
    while (j < m) {
      var st = s.st[j], k = j + 1;
      while (k < m && s.st[k] === st) k++;
      var x0 = Math.max(0, (j - 0.5) * unit), x1 = Math.min(sp.w, (k - 0.5) * unit);
      if (st === 1) { c.fillStyle = COL[fam]; c.fillRect(x0, 1, x1 - x0, mid - 2); }
      else if (st === 2) { c.fillStyle = COL.bad; c.fillRect(x0, mid + 1, x1 - x0, mid - 2); }
      else if (st === 3) { c.fillStyle = COL['ink-3']; c.globalAlpha = 0.45; c.fillRect(x0, mid - 2.5, x1 - x0, 5); c.globalAlpha = 1; }
      j = k;
    }
  }
  // each frame: the strip up to the playhead, the baseline after it
  function drawStrips() {
    shown().forEach(function (row) {
      var sp = row.strip;
      if (!sp || !sp.w) return;
      var c = sp.cv.getContext('2d'), W = sp.cv.width, H = sp.cv.height, xw = Math.round(clamp(S.tau, 0, 1) * W), dpr = W / sp.w;
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, W, H);
      c.fillStyle = COL.line; c.fillRect(xw, Math.round(H / 2 - dpr / 2), W - xw, Math.max(1, Math.round(dpr)));
      if (xw > 0) c.drawImage(sp.img, 0, 0, xw, H, 0, 0, xw, H);
    });
    if (tlHead) tlHead.style.setProperty('--x', String(S.tau));
  }
  // T2: the share of the scored frames so far that are localized within the radius (the final recall at the end)
  function updateLive() {
    if (!D || D.track !== 't2') return;
    var m = npts(Q.gt), jt = Math.floor(clamp(S.tau, 0, 1) * (m - 1));
    shown().forEach(function (row) {
      var sp = row.strip, r = row.run, text;
      if (!sp) return;
      if (r.status !== 'ok') text = '–';
      else if (S.tau >= 1) text = pct(finalRecall(r));
      else { var s = states(r); text = s.sc[jt + 1] ? pct(s.ok[jt + 1] / s.sc[jt + 1]) : '–'; }
      if (sp.text !== text) { sp.val.textContent = text; sp.text = text; }
    });
  }

  // ---------------------------------------------------------------- Systems list
  function buildLegend() {
    var miss = [], used = {};
    el.sys.replaceChildren();
    ROWS.forEach(function (row) {
      row.li = null;
      if (!row.run) { miss.push(row.name + ' (' + row.why + ')'); return; }
      var li = h('li', 'rp-sys'), btn = h('button', 'rp-sys-btn'), meta = h('div', 'rp-sys-meta');
      btn.type = 'button';
      btn.setAttribute('aria-pressed', S.on[row.fam] ? 'true' : 'false');
      row.val = h('span', 'rp-sys-val');
      btn.append(swatch(row.fam), h('span', 'rp-sys-name', row.name), ' ', row.val);   // the space keeps the name readable
      li.append(btn, meta);
      li.dataset.fam = row.fam;
      li.classList.toggle('is-off', !S.on[row.fam]);
      li.classList.toggle('is-hot', row.fam === S.hover);
      fillMeta(row, meta, used);
      btn.addEventListener('click', function () {
        S.on[row.fam] = !S.on[row.fam];
        btn.setAttribute('aria-pressed', S.on[row.fam] ? 'true' : 'false');
        li.classList.toggle('is-off', !S.on[row.fam]);
        buildStrips(); updateTexts(); renderTable(); requestDraw();
      });
      li.addEventListener('mouseenter', function () { setHover(row.fam); });
      li.addEventListener('mouseleave', function () { setHover(null); });
      btn.addEventListener('focus', function () { setHover(row.fam); });
      btn.addEventListener('blur', function () { setHover(null); });
      row.li = li;
      el.sys.appendChild(li);
    });
    el.notrun.textContent = miss.length ? 'Not run in this setup: ' + miss.join(', ') + '.' : '';
    fillNotes(used);
    fillValues();
  }
  function tag(def) { return h('span', 'rp-tag' + (def.bad ? ' rp-tag-bad' : ''), def.text); }
  function modeTag(mode) {
    var a = h('a', 'rp-tag rp-tag-mode', mode);
    a.href = '../docs/#modes';
    return a;
  }
  function fillMeta(row, meta, used) {
    var r = row.run, add = function (k) { used[k] = true; meta.appendChild(tag(TAGS[k])); };
    if (row.opts.length > 1) {
      var sel = h('select', 'rp-cfg');
      sel.setAttribute('aria-label', row.name + ' configuration');
      row.opts.forEach(function (o) {
        var op = h('option', null, o.cfg || INPUT_WORD[o.group]);
        op.value = o.system;
        op.selected = o === r;
        sel.appendChild(op);
      });
      sel.addEventListener('change', function () {
        S.cfg[row.fam + '|' + S.setup + '|' + S.inputs] = sel.value;
        buildRows(); buildLegend(); buildStrips(); updateTexts(); renderTable(); requestDraw();
        var again = el.sys.querySelector('li[data-fam="' + row.fam + '"] select');
        if (again) again.focus();
      });
      meta.appendChild(sel);
    } else if (r.cfg) {
      meta.appendChild(h('span', null, r.cfg));
    }
    if (MODES[r.system]) { used.mode = used.mode || MODES[r.system]; meta.appendChild(modeTag(MODES[r.system])); }
    if (D.track === 't1') {
      if (r.align === 'sim3') add('sim3');
      if (r.failed) add('failed');
      if (!r.e) meta.appendChild(h('span', null, 'no trajectory'));
      else if ((r.completeness || 0) < 0.995) meta.appendChild(h('span', null, 'tracked ' + pct(r.completeness || 0)));
    } else {
      if (r.status !== 'ok') add('crashed');
      if (r.upper) add('upper');
      if (r.stream) add('stream');
    }
  }
  // what the tags in the list mean, for the tags it shows
  function fillNotes(used) {
    el.notes.replaceChildren.apply(el.notes, Object.keys(TAGS).filter(function (k) { return used[k]; }).map(function (k) {
      var p = h('p', 'rp-note');
      p.append(TAGS[k].mode ? modeTag(used.mode) : tag(TAGS[k]), document.createTextNode(' ' + TAGS[k].note));
      return p;
    }));
  }
  // the value column: T1 the run's ATE, T2 its final recall within the radius
  function fillValues() {
    var t1 = D.track === 't1';
    el.metric.textContent = t1 ? 'ATE · lower is better' : 'Final recall within ' + D.thresholds[S.radius] + ' m';
    ROWS.forEach(function (row) {
      if (!row.li) return;
      var r = row.run;
      row.val.textContent = t1 ? (r.e ? metres(r.ate) : '–') : r.status !== 'ok' ? '–' : pct(finalRecall(r));
    });
  }
  function setHover(fam) {
    if (S.hover === fam) return;
    S.hover = fam;
    ROWS.forEach(function (r) { if (r.li) r.li.classList.toggle('is-hot', r.fam === fam); });
    el.tlRows.classList.toggle('is-dim', shown().some(function (r) { return r.fam === fam; }));
    [].forEach.call(el.tlRows.children, function (d) { d.classList.toggle('is-hot', d.dataset.fam === fam); });
    updateRouteKey();
    requestDraw();
  }

  // ---------------------------------------------------------------- keys, the result line, texts
  var KEY_GT = '<svg viewBox="0 0 22 12" aria-hidden="true"><path d="M1 6h20" stroke="var(--c-gt)" stroke-width="1.5" stroke-dasharray="4 3" fill="none"/></svg>';
  var KEY_NOW = '<svg viewBox="0 0 22 12" aria-hidden="true"><circle cx="11" cy="6" r="4" fill="none" stroke="var(--ink)" stroke-width="2"/></svg>';
  var KEY_MAP = '<svg viewBox="0 0 22 12" aria-hidden="true"><path d="M1 6h20" stroke="var(--ink)" stroke-opacity=".07" stroke-width="10" stroke-linecap="round"/><path d="M1 6h20" stroke="var(--ink-3)" stroke-width="1.5"/></svg>';
  var KEY_ROUTE = '<svg viewBox="0 0 22 12" aria-hidden="true"><path d="M2 6h18" stroke="var(--c)" stroke-width="3.5" stroke-linecap="round"/></svg>';
  var KEY_UP = '<svg viewBox="0 0 16 12" aria-hidden="true"><rect x="2" y="0" width="12" height="5.5" fill="var(--ink-2)"/><rect x="0" y="5.5" width="16" height="1" fill="var(--line-2)"/></svg>';
  var KEY_DOWN = '<svg viewBox="0 0 16 12" aria-hidden="true"><rect x="0" y="5.5" width="16" height="1" fill="var(--line-2)"/><rect x="2" y="6.5" width="12" height="5.5" fill="var(--bad)"/></svg>';
  var KEY_NONE = '<svg viewBox="0 0 16 12" aria-hidden="true"><rect x="2" y="3.5" width="12" height="5" fill="var(--ink-3)" fill-opacity=".45"/></svg>';
  var KEY_SKIP = '<svg viewBox="0 0 16 12" aria-hidden="true"><rect x="0" y="5.5" width="16" height="1" fill="var(--line-2)"/></svg>';
  var KEY_TRACK = '<svg viewBox="0 0 16 12" aria-hidden="true"><rect x="0" y="2" width="16" height="8" rx="4" fill="var(--ink-2)"/></svg>';
  var KEY_LOST = '<svg viewBox="0 0 16 12" aria-hidden="true"><path d="M4 2l8 8M12 2l-8 8" stroke="var(--ink-2)" stroke-width="2" stroke-linecap="round"/></svg>';
  function keyItem(tagName, icon, text) {
    var it = h(tagName);
    it.innerHTML = icon;                                // constant markup only
    it.appendChild(document.createTextNode(text));
    return it;
  }
  function buildKeys() {
    var t1 = D.track === 't1', r = D.thresholds[S.radius] + ' m';
    el.routeKey = t1 ? null : h('li', 'rp-key-route');
    el.key.replaceChildren.apply(el.key, t1
      ? [keyItem('li', KEY_GT, 'Ground truth'), keyItem('li', KEY_NOW, 'True position now')]
      : [keyItem('li', KEY_GT, 'Query session’s route'), keyItem('li', KEY_MAP, 'Map session (shaded: the area it covers)'),
        keyItem('li', KEY_NOW, 'True position, ' + r + ' radius'), el.routeKey]);
    el.tlKey.replaceChildren.apply(el.tlKey, t1
      ? [keyItem('span', KEY_TRACK, 'Tracking'), keyItem('span', KEY_LOST, 'Lost track'), h('span', null, 'Right: share of the session tracked')]
      : [keyItem('span', KEY_UP, 'Localized within ' + r + ' (system colour)'), keyItem('span', KEY_DOWN, 'Wrong, beyond ' + r),
        keyItem('span', KEY_NONE, 'No pose in the map'), keyItem('span', KEY_SKIP, 'Not scored: outside the map or standing still'),
        h('span', null, 'Right: localized so far')]);
    updateRouteKey();
  }
  // T2: the route is coloured by the state of one system (the hovered one, else the first shown)
  function updateRouteKey() {
    if (!el.routeKey) return;
    var row = focusRow();
    el.routeKey.hidden = !row;
    if (!row) return;
    el.routeKey.style.setProperty('--c', 'var(--c-' + row.fam + ')');
    var text = h('span', null, 'Route coloured by ');
    text.appendChild(h('strong', null, row.name));
    el.routeKey.innerHTML = KEY_ROUTE;                  // constant markup only
    el.routeKey.appendChild(text);
  }
  // the result of this view in one line: the lowest error (T1) or the best final recall (T2) among the systems that
  // have a run here (shown or not); failed runs do not count
  function updateBest() {
    var t1 = D.track === 't1', val, rows, top;
    val = function (row) { return t1 ? metres(row.run.ate) : pct(finalRecall(row.run)); };
    rows = ROWS.filter(function (row) {
      var r = row.run;
      return r && (t1 ? r.e && !r.failed && r.ate != null : r.status === 'ok');
    }).sort(function (a, b) { return t1 ? a.run.ate - b.run.ate : finalRecall(b.run) - finalRecall(a.run); });
    el.best.hidden = false;
    if (!rows.length || (!t1 && !finalRecall(rows[0].run))) {
      el.best.replaceChildren(h('span', 'rp-best-k', t1 ? 'No run tracked 80% of the session' : 'No system localized within ' +
        D.thresholds[S.radius] + ' m'));
      return;
    }
    top = rows.filter(function (row) { return val(row) === val(rows[0]); });
    // the parts are flex items; the spaces between them only matter to the text (screen readers, copying)
    el.best.replaceChildren(h('span', 'rp-best-k', (t1 ? 'Lowest error' : 'Best recall within ' + D.thresholds[S.radius] + ' m') + ':'));
    top.forEach(function (row, i) {
      if (i && i === top.length - 1) el.best.append(' ', h('span', 'rp-best-k', 'and'));
      el.best.append(' ', swatch(row.fam), h('strong', null, row.name + (i < top.length - 2 ? ',' : '')));
    });
    el.best.append(' ', h('span', 'rp-best-v', val(rows[0])));
    if (top.some(function (row) { return row.run.align === 'sim3'; })) el.best.append(' ', tag(TAGS.sim3));
    if (top.some(function (row) { return row.run.upper; })) el.best.append(' ', tag(TAGS.upper));
  }
  function updateTexts() {
    var t1 = S.mode === 't1', id = t1 ? S.t1 : S.t2, it = ITEMS[id], notes = [];
    el.intro.textContent = t1
      ? 'Each system maps the session on its own. Its final trajectory is drawn against the ground truth as the session plays.'
      : 'The query session runs from its first frame, with no initial pose, against each system’s map of the map session.';
    if (it) el.scnTitle.textContent = it.title;
    if (t1) el.scnSub.textContent = it ? it.sub : '';
    else if (scene(S.t2)) {
      el.scnSub.replaceChildren('Map: ', h('span', 'nowrap', session(scene(S.t2).map).label), ' → query: ',
        h('span', 'nowrap', session(curQuery()).label));
    }
    el.tableWhat.textContent = t1 ? 'sequence' : 'query session';
    if (!D || D.id !== id) return;
    if (S.inputs === 'camera') notes.push('Camera only: every system gets the same images and nothing else, the fairest comparison.');
    else if (S.inputs === 'imu') notes.push('+ IMU: each system adds the platform’s IMU to its images (visual-inertial).');
    else if (ODOMETRY[D.dataset]) notes.push(ODOMETRY[D.dataset]);
    if (D.dataset === 'kitti' && S.setup === 'rgbd') notes.push('RGB-D on KITTI is the left image with depth from stereo matching.');
    if (t1 && S.setup === 'mono' && S.inputs === 'camera') {
      notes.push('The monocular baselines are aligned with scale (Sim(3)); CROSS estimates metric scale and is aligned rigidly (SE(3)), as in the benchmark.');
    }
    el.context.textContent = notes.join(' ');
    updateBest();
    updateRouteKey();
    // for screen readers: what the view shows, with the numbers
    var vis = shown(), sum = el.scnTitle.textContent + '. ' + el.scnSub.textContent + '. ' + CAT.setups[S.setup] + ', ' +
      INPUT_WORD[S.inputs] + '. ';
    sum += vis.length ? vis.map(function (row) {
      var r = row.run;
      if (t1) return row.name + ': ' + (r.e ? 'ATE ' + metres(r.ate) : 'no trajectory') + (r.failed ? ', failed' : '') +
        ((r.completeness || 0) < 0.995 ? ', tracked ' + pct(r.completeness || 0) : '');
      return row.name + ': ' + (r.status !== 'ok' ? 'run failed' : 'final recall ' + pct(finalRecall(r)) + ' within ' + D.thresholds[S.radius] + ' m');
    }).join('. ') + '.' : 'No system shown.';
    el.summary.textContent = sum;
  }

  // ---------------------------------------------------------------- table view: every run of the sequence / query session
  function renderTable() {
    if (!el.table.open || !D) return;
    var t1 = D.track === 't1', fams = Object.keys(CAT.families), inView = {}, thead = h('thead'), tr = h('tr');
    shown().forEach(function (row) { inView[row.run.system + '|' + row.run.setup] = true; });
    var cols = t1 ? ['System and configuration', 'ATE', 'Tracked', 'Alignment']
      : ['System and configuration', 'Recall @ ' + D.thresholds[0] + ' m', 'Recall @ ' + D.thresholds[1] + ' m', 'With a pose'];
    cols.forEach(function (c, i) { var th = h('th', i ? 'n' : null, c); th.scope = 'col'; tr.appendChild(th); });
    thead.appendChild(tr);
    // one group per sensor, the chosen sensor first
    var bodies = [S.setup].concat(SETUPS.filter(function (s) { return s !== S.setup; })).map(function (setup) {
      var runs = runsNow().filter(function (r) { return r.setup === setup; });
      if (!runs.length) return null;
      runs.sort(function (a, b) {
        return fams.indexOf(a.fam) - fams.indexOf(b.fam) || GROUPS.indexOf(a.group) - GROUPS.indexOf(b.group) || a.order - b.order;
      });
      var tb = h('tbody'), gr = h('tr', 'rp-tr-group'), gh = h('th', null, CAT.setups[setup]);
      gh.colSpan = cols.length; gh.scope = 'rowgroup';
      gr.appendChild(gh); tb.appendChild(gr);
      runs.forEach(function (r) {
        // the first cell, as on the leaderboard: the system, its configuration under it
        var row = h('tr', inView[r.system + '|' + r.setup] ? 'is-shown' : null), sys = h('th', 'rp-td-sys');
        var name = h('span', 'rp-td-name'), cfg = h('span', 'rp-td-cfg', r.cfg || INPUT_WORD[r.group]);
        row.dataset.fam = r.fam;
        row.style.setProperty('--c', 'var(--c-' + r.fam + ')');
        sys.scope = 'row';
        name.append(swatch(r.fam), document.createTextNode(CAT.families[r.fam] || r.system));
        if (MODES[r.system]) cfg.append(' ', modeTag(MODES[r.system]));
        sys.append(name, cfg);
        var cells = t1
          ? [r.e ? metres(r.ate) : '–', pct(r.completeness || 0) + (r.failed ? ' (failed)' : ''), r.align === 'sim3' ? 'Sim(3)' : 'SE(3)']
          : r.status !== 'ok' ? ['run failed', '', ''] : [pct((r.lr || [])[0] || 0), pct((r.lr || [])[1] || 0), pct(r.est_frac || 0)];
        row.appendChild(sys);
        cells.forEach(function (c) { row.appendChild(h('td', 'n', c)); });
        tb.appendChild(row);
      });
      return tb;
    }).filter(Boolean);
    el.tableEl.replaceChildren.apply(el.tableEl, [thead].concat(bodies));
  }

  // ---------------------------------------------------------------- tooltip
  var tip = h('div', 'chart-tooltip');
  tip.hidden = true;
  document.body.appendChild(tip);
  function showTip(ev, nodes) {
    tip.replaceChildren.apply(tip, nodes);
    tip.hidden = false;
    var w = tip.offsetWidth, hh = tip.offsetHeight, x = ev.clientX + 14, y = ev.clientY + 16;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + hh > window.innerHeight - 8) y = ev.clientY - hh - 12;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function tipRow(fam, value, name) {
    var r = h('div', 'rp-tip-row'), k = h('i', 'rp-tip-k');
    k.style.setProperty('--c', 'var(--c-' + fam + ')');
    r.append(k, h('b', null, value), h('span', null, name));
    return r;
  }
  function tipAt(ev, j) {
    var m = npts(Q.gt), nodes = [h('div', 'rp-tip-h', pct(j / Math.max(1, m - 1)) + ' through the session' +
      (scored(Q)[j] ? '' : ' · not scored here'))];
    shown().forEach(function (row) { nodes.push(tipRow(row.fam, row.run.status !== 'ok' ? 'run failed' : stateText(states(row.run), j), row.name)); });
    showTip(ev, nodes);
  }
  function tipRun(ev, row) {
    var r = row.run, nodes = [tipRow(row.fam, r.e ? 'ATE ' + metres(r.ate) : 'no trajectory', row.name)];
    if (r.cfg) nodes.push(h('div', 'rp-tip-sub', r.cfg));
    nodes.push(h('div', 'rp-tip-sub', 'tracked ' + pct(r.completeness || 0) + ' of the session' + (r.failed ? ' · failed' : '')));
    showTip(ev, nodes);
  }
  function onStageMove(ev) {
    if (!D) return;
    var rect = el.canvas.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top, best = null, bd = 196;
    if (D.track === 't1') {
      shown().forEach(function (row) {
        var r = row.run;
        if (!r.e) return;
        var hd = headOf(r, timingOf(r), S.tau);
        if (!hd) return;
        for (var i = 0; i <= hd.i; i++) {
          var dx = X(r.e[2 * i]) - x, dy = Y(r.e[2 * i + 1]) - y, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = row; }
        }
      });
      setHover(best ? best.fam : null);
      if (best) tipRun(ev, best); else hideTip();
    } else {
      var g = Q.gt;
      for (var j = 0; j < npts(g); j++) {
        var ex = X(g[2 * j]) - x, ey = Y(g[2 * j + 1]) - y, dd = ex * ex + ey * ey;
        if (dd < bd) { bd = dd; best = j; }
      }
      if (best != null) tipAt(ev, best); else hideTip();
    }
  }

  // ---------------------------------------------------------------- playback
  var raf = 0, last = 0, scrubText = '';
  function frame(now) {
    raf = 0;
    if (S.playing) {
      if (last) S.tau = Math.min(1, S.tau + (now - last) / 1000 * S.speed / DURATION);
      last = now;
      if (S.tau >= 1) setPlaying(false);
    }
    draw();
    updateScrub();
    if (S.playing) raf = requestAnimationFrame(frame);
  }
  function requestDraw() { if (!raf) raf = requestAnimationFrame(frame); }
  function setPlaying(on) {
    S.playing = on;
    last = 0;
    el.play.classList.toggle('is-playing', on);
    el.play.setAttribute('aria-label', on ? 'Pause' : S.tau >= 1 ? 'Play again from the start' : 'Play');
    if (on) requestDraw();
  }
  function play() { if (S.tau >= 1) S.tau = 0; S.auto = false; setPlaying(true); }
  function pause() { S.auto = false; setPlaying(false); }
  function seek(tau) { S.tau = clamp(tau, 0, 1); pause(); requestDraw(); }
  function autoPause() { if (S.playing) { setPlaying(false); S.auto = true; } }
  function autoResume() { if (S.auto && S.inView && !document.hidden) { S.auto = false; setPlaying(true); } }
  // a new scenario starts from the beginning (or, with reduced motion, shows its final state)
  function restart() {
    if (reduceMotion) { S.tau = 1; setPlaying(false); return; }
    S.tau = 0;
    if (S.inView && !document.hidden) setPlaying(true); else { setPlaying(false); S.auto = true; }
  }
  function updateScrub() {
    var v = Math.round(S.tau * 1000), p = pct(S.tau);
    if (+el.scrub.value !== v) el.scrub.value = v;
    el.scrub.style.setProperty('--p', String(S.tau));
    if (scrubText !== p) { scrubText = el.pos.textContent = p; el.scrub.setAttribute('aria-valuetext', p + ' of the session'); }
  }

  // ---------------------------------------------------------------- single-choice groups (ARIA radio groups)
  // One tab stop per group; the arrow keys, Home and End move the choice; a click or Space chooses.
  function radioGroup(group, onPick) {
    group.setAttribute('role', 'radiogroup');
    group.addEventListener('click', function (e) {
      var b = e.target.closest('[role="radio"]');
      if (b && !b.disabled) onPick(b);
    });
    group.addEventListener('keydown', function (e) {
      var step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      var items = [].filter.call(group.querySelectorAll('[role="radio"]'), function (b) { return !b.disabled; });
      var i = items.indexOf(document.activeElement), n = items.length;
      if (i < 0 || (step == null && e.key !== 'Home' && e.key !== 'End')) return;
      e.preventDefault();
      var b = items[e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : (i + step + n) % n];
      b.focus();
      onPick(b);
    });
  }
  // mark the chosen option (aria-checked; .on is the shared selected style of .seg); it is the group's tab stop
  function markGroup(group, isOn) {
    var items = [].slice.call(group.querySelectorAll('[role="radio"]')), any = false;
    items.forEach(function (b) {
      var on = !!isOn(b);
      any = any || on;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.classList.toggle('on', on);
      b.tabIndex = on ? 0 : -1;
    });
    var first = items.filter(function (b) { return !b.disabled; })[0];
    if (!any && first) first.tabIndex = 0;
  }
  // on phones a row of chips scrolls sideways: keep the chosen chips in sight (again once the web font is in)
  function reveal() {
    (S.mode === 't1' ? [el.list1] : [el.list2, el.chips]).forEach(function (group) {
      var b = group.querySelector('[aria-checked="true"]');
      if (!b || group.scrollWidth <= group.clientWidth) return;
      var gb = group.getBoundingClientRect(), bb = b.getBoundingClientRect();
      if (bb.left < gb.left + 16 || bb.right > gb.right - 16) group.scrollLeft += bb.left - gb.left - (gb.width - bb.width) / 2;
    });
  }

  // ---------------------------------------------------------------- scenario selection
  // the page's lists (title, subtitle) become rows of chips; entries without data are dropped
  function readItems() {
    [el.list1, el.list2].forEach(function (list) {
      [].forEach.call(list.children, function (li) {
        var t = li.querySelector('.rp-item-t'), s = li.querySelector('.rp-item-s');
        ITEMS[li.dataset.id] = { title: (t || li).textContent.trim(), sub: s ? s.textContent.trim() : '' };
      });
    });
  }
  function buildPicker() {
    [[el.list1, CAT.mapping], [el.list2, CAT.localization]].forEach(function (p) {
      var list = p[0], known = {};
      p[1].forEach(function (it) { known[it.id] = true; });
      [].slice.call(list.children).forEach(function (li) {
        var id = li.dataset.id, b;
        if (!known[id]) { li.remove(); return; }
        b = h('button', 'rp-chip', ITEMS[id].title);
        b.type = 'button';
        b.setAttribute('role', 'radio');
        b.dataset.id = id;
        li.setAttribute('role', 'none');
        li.replaceChildren(b);
      });
      radioGroup(list, function (b) { pickScenario(b.dataset.id); });
    });
  }
  function firstId(list) { var b = list.querySelector('[role="radio"]'); return b ? b.dataset.id : null; }
  function markPicker() {
    markGroup(el.list1, function (b) { return b.dataset.id === S.t1; });
    markGroup(el.list2, function (b) { return b.dataset.id === S.t2; });
    reveal();
  }
  function pickScenario(id) {
    if (id === (S.mode === 't1' ? S.t1 : S.t2)) return;
    if (S.mode === 't1') S.t1 = id; else S.t2 = id;
    choose(true);
  }
  // the query sessions of the chosen scene (rebuilt only when the scene changes)
  function buildChips() {
    var it = scene(S.t2), cur = curQuery();
    if (!it) return;
    if (el.chips.dataset.scene !== it.id) {
      el.chips.dataset.scene = it.id;
      el.mapName.textContent = capital(session(it.map).label);
      el.chips.replaceChildren.apply(el.chips, it.queries.map(function (q) {
        var s = session(q), b = h('button', 'rp-chip', s.chip);
        b.type = 'button';
        b.setAttribute('role', 'radio');
        b.dataset.q = q;
        if (s.date) { b.title = s.label; b.setAttribute('aria-label', s.label); }  // the date shows in the viewer's subtitle
        return b;
      }));
    }
    markGroup(el.chips, function (b) { return b.dataset.q === cur; });
    reveal();
  }
  function pickQuery(q) {
    if (q === curQuery()) return;
    S.query[S.t2] = q;
    buildChips();
    if (D && D.id === S.t2) { Q = queryOf(D, q); onData(); }
    else updateTexts();
    writeHash();
  }
  function setMode(mode, user) {
    S.mode = mode;
    el.tabs.forEach(function (t) {
      var on = t.dataset.mode === mode;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    });
    el.picks.forEach(function (p) { p.classList.toggle('is-active', p.dataset.mode === mode); });
    el.panel.setAttribute('aria-labelledby', 'rp-tab-' + mode);
    el.radiusFilter.hidden = mode !== 't2';
    choose(user);
  }
  function choose(user) {
    var id = S.mode === 't1' ? S.t1 : S.t2, seq = ++loadSeq;
    markPicker();
    if (S.mode === 't2') buildChips();
    updateTexts();
    if (user) writeHash();
    el.viewer.classList.add('is-loading');
    load(id).then(function (d) {
      if (seq !== loadSeq) return;
      D = d;
      Q = d.track === 't2' ? queryOf(d, curQuery()) : null;
      el.msg.hidden = true;
      onData();
    }, function () {
      if (seq !== loadSeq) return;
      el.msg.textContent = 'This replay could not be loaded. The numbers are in the benchmark summary.';
      el.msg.hidden = false;
      el.viewer.classList.remove('is-loading');
    });
  }
  // after the scenario (or the query session) changed
  function onData() {
    el.viewer.classList.remove('is-loading');
    availability();
    markFilters();
    buildRows(); buildLegend(); buildKeys(); buildStrips();
    refit(); paintStatic();
    updateTexts(); renderTable();
    restart();
    requestDraw();
  }
  // after a filter changed (same scenario: the playhead keeps its place)
  function onFilter() {
    availability();
    markFilters();
    buildRows(); buildLegend(); buildKeys(); buildStrips();
    updateTexts(); renderTable(); requestDraw();
    writeHash();
  }
  // a sensor without runs in this scenario, or inputs without runs with the chosen sensor, cannot be chosen; when the
  // chosen one has none, the first that has runs takes its place
  function availability() {
    var runs = runsNow(), has = function (setup, group) {
      return runs.some(function (r) { return r.setup === setup && (!group || r.group === group); });
    };
    [].forEach.call(el.setup.children, function (b) { b.disabled = !has(b.dataset.setup); });
    if (!has(S.setup)) S.setup = SETUPS.filter(function (s) { return has(s); })[0] || S.setup;
    [].forEach.call(el.inputs.children, function (b) { b.disabled = !has(S.setup, b.dataset.inputs); });
    if (!has(S.setup, S.inputs)) S.inputs = GROUPS.filter(function (g) { return has(S.setup, g); })[0] || S.inputs;
  }
  function markFilters() {
    markGroup(el.setup, function (b) { return b.dataset.setup === S.setup; });
    markGroup(el.inputs, function (b) { return b.dataset.inputs === S.inputs; });
    markGroup(el.speed, function (b) { return +b.dataset.speed === S.speed; });
    if (!D || D.track !== 't2') return;
    // the success radii of this dataset (rebuilt only when they change)
    var key = D.thresholds.join('|');
    if (el.radius.dataset.key !== key) {
      el.radius.dataset.key = key;
      el.radius.replaceChildren.apply(el.radius, D.thresholds.map(function (t, i) {
        var b = h('button', null, t + ' m');
        b.type = 'button';
        b.setAttribute('role', 'radio');
        b.dataset.i = i;
        return b;
      }));
    }
    markGroup(el.radius, function (b) { return +b.dataset.i === S.radius; });
  }

  // ---------------------------------------------------------------- URL hash: #m=<sequence> or #l=<scene>&q=<session>, filters
  function readHash() {
    var p = new URLSearchParams(location.hash.slice(1));
    var has = function (list, id) { return !!id && [].some.call(list.querySelectorAll('[role="radio"]'), function (b) { return b.dataset.id === id; }); };
    if (has(el.list1, p.get('m'))) { S.mode = 't1'; S.t1 = p.get('m'); }
    if (has(el.list2, p.get('l'))) { S.mode = 't2'; S.t2 = p.get('l'); if (p.get('q')) S.query[S.t2] = p.get('q'); }
    if (SETUPS.indexOf(p.get('setup')) >= 0) S.setup = p.get('setup');
    if (GROUPS.indexOf(p.get('in')) >= 0) S.inputs = p.get('in');
    if (p.get('r') === '0' || p.get('r') === '1') S.radius = +p.get('r');
  }
  function writeHash() {
    var p = new URLSearchParams();
    if (S.mode === 't1') p.set('m', S.t1); else { p.set('l', S.t2); p.set('q', curQuery()); p.set('r', String(S.radius)); }
    p.set('setup', S.setup);
    p.set('in', S.inputs);
    try { history.replaceState(null, '', '#' + p.toString()); } catch (e) {}
  }

  // ---------------------------------------------------------------- events
  function bind() {
    el.tabs.forEach(function (t, i) {
      t.addEventListener('click', function () { if (S.mode !== t.dataset.mode) setMode(t.dataset.mode, true); });
      t.addEventListener('keydown', function (e) {
        var n = el.tabs.length, k = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: n - 1 }[e.key];
        if (k == null) return;
        e.preventDefault();
        var next = el.tabs[(k + n) % n];
        next.focus();
        if (S.mode !== next.dataset.mode) setMode(next.dataset.mode, true);
      });
    });
    radioGroup(el.chips, function (b) { pickQuery(b.dataset.q); });
    radioGroup(el.setup, function (b) { if (S.setup !== b.dataset.setup) { S.setup = b.dataset.setup; onFilter(); } });
    radioGroup(el.inputs, function (b) { if (S.inputs !== b.dataset.inputs) { S.inputs = b.dataset.inputs; onFilter(); } });
    radioGroup(el.radius, function (b) { if (S.radius !== +b.dataset.i) { S.radius = +b.dataset.i; onFilter(); } });
    radioGroup(el.speed, function (b) { S.speed = +b.dataset.speed; markFilters(); });
    el.play.addEventListener('click', function () { if (S.playing) pause(); else play(); });
    el.restart.addEventListener('click', function () { S.tau = 0; play(); });
    el.scrub.addEventListener('input', function () { seek(+el.scrub.value / 1000); });
    el.scrub.addEventListener('keydown', function (e) { if (e.key === ' ') { e.preventDefault(); if (S.playing) pause(); else play(); } });
    el.stage.addEventListener('keydown', function (e) {
      if (e.key === ' ') { e.preventDefault(); if (S.playing) pause(); else play(); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); seek(S.tau + (e.key === 'ArrowRight' ? 0.02 : -0.02)); }
      else if (e.key === 'Home' || e.key === 'End') { e.preventDefault(); seek(e.key === 'Home' ? 0 : 1); }
    });
    el.stage.addEventListener('pointermove', onStageMove);
    el.stage.addEventListener('pointerleave', function () { if (D && D.track === 't1') setHover(null); hideTip(); });
    // the strips: hover shows the states at that time, click or drag seeks
    var dragging = false;
    function stripFrac(ev) {                         // the time under the pointer, null off the strips
      var cv = el.tlRows.querySelector('.rp-tl-strip');
      if (!cv) return null;
      var r = cv.getBoundingClientRect(), f = (ev.clientX - r.left) / Math.max(1, r.width);
      return f < -0.01 || f > 1.01 ? null : clamp(f, 0, 1);
    }
    el.tlRows.addEventListener('pointerdown', function (ev) {
      var f = stripFrac(ev);
      if (f == null) return;
      dragging = true;
      if (el.tlRows.setPointerCapture) el.tlRows.setPointerCapture(ev.pointerId);
      seek(f);
    });
    el.tlRows.addEventListener('pointermove', function (ev) {
      var f = stripFrac(ev), row = ev.target.closest && ev.target.closest('.rp-tl-row');
      if (!D) return;
      if (row) setHover(row.dataset.fam);
      if (f == null) { hideTip(); return; }
      if (dragging) seek(f);
      if (D.track === 't2') tipAt(ev, Math.round(f * (npts(Q.gt) - 1)));
      else if (row) { var hit = ROWS.filter(function (r) { return r.fam === row.dataset.fam; })[0]; if (hit && hit.run) tipRun(ev, hit); }
    });
    el.tlRows.addEventListener('pointerup', function () { dragging = false; });
    el.tlRows.addEventListener('pointercancel', function () { dragging = false; });
    el.tlRows.addEventListener('pointerleave', function () { if (!dragging) { setHover(null); hideTip(); } });
    el.table.addEventListener('toggle', renderTable);
    window.addEventListener('hashchange', function () { readHash(); markFilters(); setMode(S.mode, false); });
    // pause while the tab is hidden or the viewer is off-screen
    document.addEventListener('visibilitychange', function () { if (document.hidden) autoPause(); else autoResume(); });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (en) {
        S.inView = en[en.length - 1].isIntersecting;
        if (S.inView) autoResume(); else autoPause();
      }, { threshold: 0.2 }).observe(el.stage);
    } else {
      S.inView = true;
    }
    if ('ResizeObserver' in window) {
      new ResizeObserver(resize).observe(el.stage);
      new ResizeObserver(paintStrips).observe(el.tlRows);
    } else {
      window.addEventListener('resize', function () { resize(); paintStrips(); });
    }
    function onTheme() { readColours(); paintStatic(); paintStrips(); requestDraw(); }
    document.addEventListener('cross:theme', onTheme);
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) mq.addEventListener('change', onTheme); else if (mq.addListener) mq.addListener(onTheme);
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { onTheme(); reveal(); });
  }

  // ---------------------------------------------------------------- start
  readItems();
  fetch(DATA + 'index.json').then(function (res) {
    if (!res.ok) throw new Error(res.status);
    return res.json();
  }).then(function (cat) {
    CAT = cat;
    el.panel.setAttribute('role', 'tabpanel');
    Object.keys(cat.systems).forEach(function (k, i) { ORDER[k] = i; });
    buildPicker();
    S.t1 = firstId(el.list1);
    S.t2 = firstId(el.list2);
    readColours();
    readHash();
    bind();
    markFilters();
    resize();
    setMode(S.mode, false);
  }).catch(function () {
    el.msg.textContent = 'The replay data could not be loaded. The numbers are in the benchmark summary.';
    el.msg.hidden = false;
  });
})();
