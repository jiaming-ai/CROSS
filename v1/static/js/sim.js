/*
 * Toy 2D simulation of CROSS's pre-commitment layer vs. committing to the
 * top visual match. Everything is precomputed so the UI can scrub freely.
 *
 * World: two look-alike corridors (B at y=4.5, T at y=10) joined on the left.
 * The robot is kidnapped into corridor B, drives east, then turns south.
 * VPR returns the true keyframe and its look-alike twin in corridor T with
 * similar scores, so appearance alone cannot tell them apart; only the turn
 * (which the twin corridor does not have) disambiguates them.
 */
(function (root) {
  'use strict';

  const TAU = Math.PI * 2;
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  }

  // ---------- 2x2 helpers ----------
  const m2 = {
    add: (A, B) => [A[0] + B[0], A[1] + B[1], A[2] + B[2]], // [a, b, d] symmetric [[a,b],[b,d]]
    inv: (A) => { const det = A[0] * A[2] - A[1] * A[1]; return [A[2] / det, -A[1] / det, A[0] / det]; },
    rot: (A, th) => {
      const c = Math.cos(th), s = Math.sin(th);
      // R A R^T
      const a = A[0], b = A[1], d = A[2];
      return [
        c * c * a - 2 * c * s * b + s * s * d,
        c * s * a + (c * c - s * s) * b - c * s * d,
        s * s * a + 2 * c * s * b + c * c * d,
      ];
    },
    quad: (A, x, y) => A[0] * x * x + 2 * A[1] * x * y + A[2] * y * y,
    mul: (A, B) => [ // A*B (general 2x2 as [a,b,c,d]) for symmetric inputs
      A[0] * B[0] + A[1] * B[1], A[0] * B[1] + A[1] * B[2],
      A[1] * B[0] + A[2] * B[1], A[1] * B[1] + A[2] * B[2],
    ],
  };

  // ---------- world ----------
  const WORLD = { w: 24, h: 14.5 };
  const Y_B = 4.5, Y_T = 10, DY = Y_T - Y_B;

  function buildMap() {
    const nodes = [];
    const edges = [];
    const add = (x, y, region) => { nodes.push({ x, y, region }); return nodes.length - 1; };
    const chain = (ids) => { for (let i = 1; i < ids.length; i++) edges.push([ids[i - 1], ids[i]]); };
    const B = [], T = [], L = [];
    for (let x = 2; x <= 18; x++) B.push(add(x, Y_B, 'B'));
    for (let x = 2; x <= 18; x++) T.push(add(x, Y_T, 'T'));
    for (let y = 5.5; y <= 9.01; y += 1.1667) L.push(add(2, y, 'L'));
    chain(B); chain(T); chain([B[0], ...L, T[0]]);
    const BS = [B[B.length - 1]];
    for (const y of [3.5, 2.5, 1.5]) BS.push(add(18, y, 'BS'));
    for (const x of [19, 20, 21, 22]) BS.push(add(x, 1.5, 'BS'));
    chain(BS);
    const TN = [T[T.length - 1]];
    for (const y of [11, 12, 13]) TN.push(add(18, y, 'TN'));
    for (const x of [19, 20, 21, 22]) TN.push(add(x, 13, 'TN'));
    chain(TN);
    // corridor centre-lines for drawing walls
    const corridors = [
      [[2, Y_B], [18, Y_B], [18, 1.5], [22.6, 1.5]],
      [[2, Y_T], [18, Y_T], [18, 13], [22.6, 13]],
      [[2, Y_B], [2, Y_T]],
    ];
    return { nodes, edges, corridors };
  }

  function groundTruth(v) {
    const cmds = [
      { t: 'S', len: 14.2 },
      { t: 'T', ang: -Math.PI / 2, r: 0.8 },
      { t: 'S', len: 1.4 },
      { t: 'T', ang: Math.PI / 2, r: 0.8 },
      { t: 'S', len: 2.9 },
    ];
    const poses = [];
    let x = 3.0, y = Y_B, th = 0;
    poses.push([x, y, th]);
    for (const c of cmds) {
      const len = c.t === 'S' ? c.len : Math.abs(c.ang) * c.r;
      const n = Math.max(1, Math.round(len / v));
      const ds = len / n;
      for (let i = 0; i < n; i++) {
        if (c.t === 'S') { x += ds * Math.cos(th); y += ds * Math.sin(th); }
        else {
          const dth = (c.ang / n);
          const thm = th + dth / 2;
          x += ds * Math.cos(thm); y += ds * Math.sin(thm);
          th = wrap(th + dth);
        }
        poses.push([x, y, th]);
      }
    }
    return poses;
  }

  const CROWD = { x0: 9.4, x1: 11.6 }; // no retrievals here (occluded by pedestrians)

  function nearestNodes(map, x, y, k, maxD, filter) {
    const out = [];
    map.nodes.forEach((n, i) => {
      if (filter && !filter(n)) return;
      const d = Math.hypot(n.x - x, n.y - y);
      if (d <= maxD) out.push({ i, d });
    });
    out.sort((a, b) => a.d - b.d);
    return out.slice(0, k);
  }

  function simulate(seed) {
    const rng = mulberry32(seed);
    const map = buildMap();
    const gt = groundTruth(0.1);
    const N = gt.length;
    const RETRIEVE_EVERY = 3;

    // odometry with noise and a slow heading bias
    const bias = (rng() < 0.5 ? -1 : 1) * (0.0012 + 0.0008 * rng());
    const odo = [null];
    for (let t = 1; t < N; t++) {
      const [x0, y0, a0] = gt[t - 1], [x1, y1, a1] = gt[t];
      const dx = x1 - x0, dy = y1 - y0;
      const c = Math.cos(a0), s = Math.sin(a0);
      let fx = c * dx + s * dy, fy = -s * dx + c * dy, da = wrap(a1 - a0);
      fx *= 1 + 0.05 * gauss(rng);
      fy += 0.004 * gauss(rng);
      da += bias + 0.01 * gauss(rng);
      odo.push([fx, fy, da]);
    }

    // retrievals: {kf, score, z:[x,y,th], kind:'true'|'alias'|'distractor'}
    const retr = new Array(N).fill(null);
    for (let t = 2; t < N; t += RETRIEVE_EVERY) {
      const [x, y, th] = gt[t];
      if (y > 4 && x > CROWD.x0 && x < CROWD.x1) { retr[t] = []; continue; }
      const cands = [];
      const near = nearestNodes(map, x, y, 2, 1.4, (n) => n.region !== 'T' && n.region !== 'TN');
      near.forEach((nn, j) => {
        const score = 0.52 + 0.3 * rng() - 0.08 * j;
        cands.push({ kf: nn.i, score, kind: 'true',
          z: [x + 0.13 * gauss(rng), y + 0.13 * gauss(rng), wrap(th + 0.03 * gauss(rng))] });
      });
      const onAliasedStretch = Math.abs(y - Y_B) < 0.05 && x >= 2 && x <= 17.2;
      if (onAliasedStretch) {
        const twin = nearestNodes(map, x, y + DY, 2, 1.4, (n) => n.region === 'T');
        twin.forEach((nn, j) => {
          const score = 0.52 + 0.3 * rng() - 0.08 * j;
          cands.push({ kf: nn.i, score, kind: 'alias',
            z: [x + 0.13 * gauss(rng), y + DY + 0.13 * gauss(rng), wrap(th + 0.03 * gauss(rng))] });
        });
      }
      if (rng() < 0.12) {
        // a spurious retrieval from somewhere unrelated
        const pool = map.nodes.map((n, i) => ({ n, i })).filter(({ n }) =>
          Math.hypot(n.x - x, n.y - y) > 4 && Math.hypot(n.x - x, n.y - y - DY) > 4);
        const pick = pool[Math.floor(rng() * pool.length)];
        const a = rng() * TAU;
        cands.push({ kf: pick.i, score: 0.5 + 0.32 * rng(), kind: 'distractor',
          z: [pick.n.x + 0.3 * Math.cos(a), pick.n.y + 0.3 * Math.sin(a), wrap(rng() * TAU)] });
      }
      retr[t] = cands;
    }

    // ---------- baseline: commit to the top-scoring retrieval ----------
    const greedy = new Array(N).fill(null);
    const falseEdges = []; // {t, a:[x,y], b:[x,y]}
    let g = null, wrongCommits = 0, commits = 0;
    for (let t = 0; t < N; t++) {
      if (g && t > 0) g = compose(g, odo[t]);
      const cands = retr[t];
      let event = null;
      if (cands && cands.length) {
        const best = cands.reduce((a, b) => (b.score > a.score ? b : a));
        const prev = g;
        g = best.z.slice();
        commits++;
        const err = Math.hypot(g[0] - gt[t][0], g[1] - gt[t][1]);
        if (err > 2) {
          wrongCommits++;
          const kfTrue = nearestNodes(map, gt[t][0], gt[t][1], 1, 3, (n) => n.region !== 'T' && n.region !== 'TN')[0];
          const kfA = map.nodes[best.kf];
          if (kfTrue) falseEdges.push({ t, a: [map.nodes[kfTrue.i].x, map.nodes[kfTrue.i].y], b: [kfA.x, kfA.y] });
          event = { kind: 'wrong', kf: best.kf };
        } else event = { kind: 'ok', kf: best.kf };
        if (prev && Math.hypot(prev[0] - g[0], prev[1] - g[1]) > 2) event.jump = true;
      }
      greedy[t] = { est: g ? g.slice() : null, event, wrongCommits, commits,
        err: g ? Math.hypot(g[0] - gt[t][0], g[1] - gt[t][1]) : null };
    }

    // ---------- CROSS-style bounded Gaussian-mixture filter ----------
    const Q_ALONG = 0.0035, Q_CROSS = 0.0012, Q_TH = 0.00025;
    const R_POS = 0.13 * 0.13 + 0.04, R_TH = 0.03 * 0.03 + 0.002;
    const ALPHA = 0.3;       // likelihood floor for a branch no retrieval supports
    const FADE = 0.8;
    const GATE = 9.21;       // chi^2(2) 99%
    const PRUNE = 0.03, KMAX = 5;
    const W = 8, R_NEED = 6, DOMINANT = 0.8;
    let hyps = [];
    let nextId = 1;
    let committed = null; // {id, t}
    const frames = new Array(N);
    const events = [];
    const history = {}; // id -> [{t, x, y}]

    for (let t = 0; t < N; t++) {
      // predict
      if (t > 0) {
        for (const h of hyps) {
          h.mu = compose(h.mu, odo[t]);
          const Qw = m2.rot([Q_ALONG, 0, Q_CROSS], h.mu[2]);
          h.P = m2.add(h.P, Qw);
          h.Pth += Q_TH;
        }
      }
      let modes = null;
      const cands = retr[t];
      if (cands && cands.length) {
        // association weights: softmax over VPR score x (simulated) inlier ratio
        const beta = 4;
        const ex = cands.map((c) => Math.exp(beta * c.score));
        const sum = ex.reduce((a, b) => a + b, 0);
        const P = ex.map((e) => e / sum);
        // cluster measurement components in pose space (DBSCAN-like, eps = 1 m)
        modes = [];
        cands.forEach((c, i) => {
          let m = modes.find((mm) => Math.hypot(mm.sx / mm.w - c.z[0], mm.sy / mm.w - c.z[1]) < 1.0);
          if (!m) { m = { z: [0, 0, 0], w: 0, kfs: [], kinds: [], sx: 0, sy: 0, sc: 0, ss: 0 }; modes.push(m); }
          m.w += P[i]; m.kfs.push(c.kf); m.kinds.push(c.kind);
          m.sx += P[i] * c.z[0]; m.sy += P[i] * c.z[1];
          m.sc += P[i] * Math.cos(c.z[2]); m.ss += P[i] * Math.sin(c.z[2]);
        });
        for (const m of modes) {
          m.z = [m.sx / m.w, m.sy / m.w, Math.atan2(m.ss, m.sc)];
          m.R = [R_POS, 0, R_POS];
          m.kind = m.kinds.includes('true') ? 'true' : m.kinds.includes('alias') ? 'alias' : 'distractor';
        }
        // weight update + fusion (product of motion and measurement mixtures)
        const matched = new Set();
        for (const h of hyps) {
          let L = ALPHA, best = null, bestG = 0;
          modes.forEach((m, j) => {
            const S = m2.add(h.P, m.R);
            const Si = m2.inv(S);
            const dx = m.z[0] - h.mu[0], dy = m.z[1] - h.mu[1];
            const dth = wrap(m.z[2] - h.mu[2]);
            const d2 = m2.quad(Si, dx, dy) + (dth * dth) / (h.Pth + R_TH);
            const gv = Math.exp(-0.5 * d2);
            L += m.w * gv;
            if (d2 < GATE && gv > bestG) { bestG = gv; best = j; }
          });
          // fading memory keeps look-alike branches from drifting apart on noise alone
          h.w = Math.pow(h.w, FADE) * L;
          if (best !== null) {
            const m = modes[best];
            matched.add(best);
            const Si = m2.inv(m2.add(h.P, m.R));
            const K = m2.mul(h.P, Si); // [k00,k01,k10,k11]
            const dx = m.z[0] - h.mu[0], dy = m.z[1] - h.mu[1];
            h.mu[0] += K[0] * dx + K[1] * dy;
            h.mu[1] += K[2] * dx + K[3] * dy;
            // P = (I-K)P
            const a = h.P[0], b = h.P[1], d = h.P[2];
            h.P = [
              (1 - K[0]) * a - K[1] * b,
              (1 - K[0]) * b - K[1] * d,
              -K[2] * b + (1 - K[3]) * d,
            ];
            const kth = h.Pth / (h.Pth + R_TH);
            h.mu[2] = wrap(h.mu[2] + kth * wrap(m.z[2] - h.mu[2]));
            h.Pth *= 1 - kth;
            h.supported = true;
          } else h.supported = false;
        }
        // births from modes no branch explains
        const firstBirth = hyps.length === 0;
        modes.forEach((m, j) => {
          if (matched.has(j)) return;
          const h = { id: nextId++, born: t, mu: m.z.slice(), P: m.R.slice(), Pth: R_TH,
            w: firstBirth ? m.w : 0.12 * m.w, support: [], supported: true, kind: m.kind };
          hyps.push(h);
          events.push({ t, type: 'birth', id: h.id, kind: m.kind });
        });
        // normalise
        let tot = hyps.reduce((a, h) => a + h.w, 0);
        hyps.forEach((h) => (h.w /= tot));
        // merge near-duplicate branches
        for (let i = 0; i < hyps.length; i++) for (let j = hyps.length - 1; j > i; j--) {
          const a = hyps[i], b = hyps[j];
          if (Math.hypot(a.mu[0] - b.mu[0], a.mu[1] - b.mu[1]) < 0.6 && Math.abs(wrap(a.mu[2] - b.mu[2])) < 0.4) {
            const keep = a.born <= b.born ? a : b, drop = keep === a ? b : a;
            keep.w += drop.w;
            drop.dead = { t, why: 'merged', into: keep.id };
            events.push({ t, type: 'merge', id: drop.id, into: keep.id });
            hyps.splice(hyps.indexOf(drop), 1);
            if (drop === a) { i--; break; }
          }
        }
        // prune
        hyps.sort((a, b) => b.w - a.w);
        const survivors = [];
        hyps.forEach((h, k) => {
          const isCommitted = committed && committed.id === h.id;
          if (!isCommitted && (h.w < PRUNE || k >= KMAX)) {
            events.push({ t, type: 'prune', id: h.id, kind: h.kind, w: h.w });
          } else survivors.push(h);
        });
        hyps = survivors;
        tot = hyps.reduce((a, h) => a + h.w, 0);
        hyps.forEach((h) => (h.w /= tot));
        // delayed commitment: dominant in R_NEED of the last W updates
        for (const h of hyps) {
          h.support.push(h.w >= DOMINANT);
          if (h.support.length > W) h.support.shift();
          const count = h.support.filter(Boolean).length;
          h.count = count;
          if (!committed && count >= R_NEED) {
            committed = { id: h.id, t };
            events.push({ t, type: 'commit', id: h.id, kind: h.kind });
          }
        }
      }
      for (const h of hyps) {
        (history[h.id] = history[h.id] || []).push({ t, x: h.mu[0], y: h.mu[1] });
      }
      const ch = committed ? hyps.find((h) => h.id === committed.id) : null;
      frames[t] = {
        hyps: hyps.map((h) => ({ id: h.id, born: h.born, mu: h.mu.slice(), P: h.P.slice(), w: h.w,
          support: h.support.slice(), count: h.count || 0, kind: h.kind, supported: h.supported })),
        modes: modes ? modes.map((m) => ({ z: m.z.slice(), w: m.w, kfs: m.kfs.slice(), kind: m.kind })) : null,
        committed: committed ? { ...committed } : null,
        err: ch ? Math.hypot(ch.mu[0] - gt[t][0], ch.mu[1] - gt[t][1]) : null,
      };
    }

    const commitEvent = events.find((e) => e.type === 'commit') || null;
    return {
      seed, map, gt, N, retr, greedy, falseEdges, frames, events, history, WORLD, CROWD,
      params: { W, R_NEED, DOMINANT, KMAX, RETRIEVE_EVERY },
      summary: {
        greedyWrong: greedy[N - 1].wrongCommits,
        greedyCommits: greedy[N - 1].commits,
        commitT: commitEvent ? commitEvent.t : null,
        commitKind: commitEvent ? commitEvent.kind : null,
        finalErr: frames[N - 1].err,
      },
    };
  }

  function compose(p, u) {
    const c = Math.cos(p[2]), s = Math.sin(p[2]);
    return [p[0] + c * u[0] - s * u[1], p[1] + s * u[0] + c * u[1], wrap(p[2] + u[2])];
  }

  const api = { simulate, WORLD };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CrossSim = api;
})(typeof window !== 'undefined' ? window : globalThis);
