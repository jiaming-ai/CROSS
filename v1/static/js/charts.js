/* Small SVG chart helpers for the project page. Colors come from CSS classes,
 * so every chart follows the light/dark theme without re-rendering. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, str, cls, anchor) {
    const t = el('text', { x, y, class: cls, 'text-anchor': anchor || 'start', 'dominant-baseline': 'middle' }, parent);
    t.textContent = str;
    return t;
  }
  // bar with a 4px rounded data-end, square at the baseline (horizontal)
  function hBarPath(x, y, w, h, r) {
    r = Math.min(r, h / 2, Math.max(w, 0));
    if (w <= 0) return `M${x},${y}h0v${h}h0z`;
    return `M${x},${y}h${w - r}q${r},0 ${r},${r}v${h - 2 * r}q0,${r} ${-r},${r}h${-(w - r)}z`;
  }
  // vertical bar, rounded top
  function vBarPath(x, yBase, w, h, r) {
    r = Math.min(r, w / 2, Math.max(h, 0));
    if (h <= 0) return `M${x},${yBase}h${w}v0h${-w}z`;
    return `M${x},${yBase}v${-(h - r)}q0,${-r} ${r},${-r}h${w - 2 * r}q${r},0 ${r},${r}v${h - r}z`;
  }

  // ------------------------------------------------------------ tooltip
  const tip = {
    node: null,
    show(evt, lines, anchorRect) {
      if (!this.node) this.node = document.getElementById('tooltip');
      const n = this.node;
      n.replaceChildren();
      lines.forEach(([cls, str]) => {
        const d = document.createElement('div');
        d.className = cls;
        d.textContent = str;
        n.appendChild(d);
      });
      n.hidden = false;
      let x, y;
      if (evt && evt.clientX !== undefined && evt.type !== 'focus') { x = evt.clientX + 14; y = evt.clientY + 14; }
      else if (anchorRect) { x = anchorRect.right + 8; y = anchorRect.top; }
      const r = n.getBoundingClientRect();
      if (x + r.width > window.innerWidth - 8) x = (evt && evt.clientX ? evt.clientX : x) - r.width - 14;
      if (y + r.height > window.innerHeight - 8) y = window.innerHeight - r.height - 8;
      n.style.left = Math.max(8, x) + 'px';
      n.style.top = Math.max(8, y) + 'px';
    },
    hide() { if (this.node) this.node.hidden = true; },
  };

  function tween(from, to, dur, step) {
    if (reduceMotion || dur === 0) { step(to, 1); return; }
    const t0 = performance.now();
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    function frame(now) {
      const k = Math.min(1, (now - t0) / dur);
      const e = ease(k);
      step(from.map((f, i) => f + (to[i] - f) * e), k);
      if (k < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function observeWidth(node, cb) {
    let last = node.clientWidth;
    const ro = new ResizeObserver(() => {
      const w = node.clientWidth;
      if (Math.abs(w - last) > 2) { last = w; cb(); }
    });
    ro.observe(node);
  }

  // legend that wraps onto a new row when it runs out of width; returns its height
  function legend(svg, items, x0, y0, maxW) {
    let x = x0, y = y0;
    items.forEach((it) => {
      const g = el('g', {}, svg);
      if (it.kind === 'line') el('line', { x1: 0, x2: 18, y1: 0, y2: 0, stroke: it.stroke, 'stroke-width': 2.5 }, g);
      else if (it.kind === 'dot') el('circle', { cx: 6, cy: 0, r: 5, class: it.cls }, g);
      else el('rect', { x: 0, y: -6, width: 12, height: 12, rx: 3, class: it.cls }, g);
      const off = it.kind === 'line' ? 24 : 18;
      const t = text(g, off, 0.5, it.label, 'c-legend');
      const w = off + t.getComputedTextLength();
      if (x > x0 && x + w > x0 + maxW) { x = x0; y += 20; }
      g.setAttribute('transform', `translate(${x},${y})`);
      x += w + 18;
    });
    return y - y0 + 12;
  }

  // ------------------------------------------------------------ horizontal bars
  /* spec: {rows:[{key,label,group?,hl?}], values:{key:{v, ci?:[lo,hi], n?}}, max, ticks:[], fmt(v), tickFmt(v), unitNote} */
  function HBar(container, spec) {
    this.c = container;
    this.spec = spec;
    this.cur = null;
    this.build();
    observeWidth(container, () => { this.build(); this.update(this.spec.values, 0); });
  }
  HBar.prototype.build = function () {
    const s = this.spec, c = this.c;
    c.replaceChildren();
    const W = Math.max(280, c.clientWidth || 600);
    const narrow = W < 460;
    const labelW = narrow ? 92 : 128;
    const right = narrow ? 64 : 72;
    const rowH = narrow ? 28 : 30, barH = narrow ? 13 : 15, groupH = 26, top = 24;
    let y = top;
    const layout = [];
    let lastGroup = null;
    s.rows.forEach((r) => {
      if (r.group && r.group !== lastGroup) { layout.push({ kind: 'group', label: r.group, y: y + 12 }); y += groupH; lastGroup = r.group; }
      layout.push({ kind: 'row', row: r, y });
      y += rowH;
    });
    const H = y + 6;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': s.aria || '' }, c);
    const x0 = labelW, x1 = W - right;
    this.geom = { W, H, x0, x1, barH, rowH };
    const scale = (v) => x0 + (x1 - x0) * Math.max(0, Math.min(1, v / s.max));
    this.scale = scale;
    // grid + ticks
    const g = el('g', {}, svg);
    const ticks = narrow && s.ticks.length > 4 ? s.ticks.filter((_, i) => i % 2 === 0) : s.ticks;
    ticks.forEach((t) => {
      const x = scale(t);
      el('line', { x1: x, x2: x, y1: top - 6, y2: H - 4, class: t === 0 ? 'c-axis' : 'c-grid' }, g);
      text(g, x, top - 14, s.tickFmt(t), 'c-tick', 'middle');
    });
    this.rows = {};
    layout.forEach((it) => {
      if (it.kind === 'group') { text(svg, 0, it.y, it.label, 'c-group'); return; }
      const r = it.row;
      const rg = el('g', { class: 'c-row', tabindex: 0, role: 'listitem' }, svg);
      const cy = it.y + rowH / 2;
      text(rg, x0 - 12, cy, r.label, 'c-label' + (r.hl ? ' c-label-hl' : ''), 'end');
      const bar = el('path', { class: r.hl ? 'c-bar-hl' : (r.alt ? 'c-bar-2' : 'c-bar') }, rg);
      const ci = el('g', { class: 'c-ci-g' }, rg);
      const ciLine = el('line', { class: 'c-ci', y1: cy, y2: cy }, ci);
      const ciA = el('line', { class: 'c-ci', y1: cy - 4, y2: cy + 4 }, ci);
      const ciB = el('line', { class: 'c-ci', y1: cy - 4, y2: cy + 4 }, ci);
      const val = text(rg, 0, cy, '', 'c-value' + (r.hl ? ' c-value-hl' : ''));
      const hit = el('rect', { x: 0, y: it.y, width: W, height: rowH, class: 'c-hit' }, rg);
      rg.insertBefore(hit, rg.firstChild);
      const show = (e) => {
        const d = this.spec.values[r.key];
        if (!d) return;
        const lines = [['tt-v', s.fmt(d.v)], ['tt-n', r.label]];
        if (d.ci) lines.push(['tt-s', `95% CI ${s.fmt(d.ci[0] / (s.ciScale || 1))} – ${s.fmt(d.ci[1] / (s.ciScale || 1))}`]);
        if (d.n) lines.push(['tt-s', `n = ${d.n.toLocaleString()} trials`]);
        tip.show(e, lines, bar.getBoundingClientRect());
      };
      rg.addEventListener('pointermove', show);
      rg.addEventListener('focus', show);
      rg.addEventListener('pointerleave', () => tip.hide());
      rg.addEventListener('blur', () => tip.hide());
      this.rows[r.key] = { y: it.y, cy, bar, ci, ciLine, ciA, ciB, val };
    });
    this.cur = null;
  };
  HBar.prototype.update = function (values, dur) {
    this.spec.values = values;
    const keys = this.spec.rows.map((r) => r.key);
    const target = [];
    keys.forEach((k) => {
      const d = values[k] || { v: 0 };
      const cs = this.spec.ciScale || 1;
      target.push(d.v, d.ci ? d.ci[0] / cs : d.v, d.ci ? d.ci[1] / cs : d.v, d.ci ? 1 : 0);
    });
    const from = this.cur || target.map((v, i) => (i % 4 === 3 ? v : 0));
    const { barH, x0 } = this.geom;
    const s = this.spec;
    tween(from, target, dur === undefined ? 450 : dur, (vals) => {
      this.cur = vals;
      keys.forEach((k, i) => {
        const R = this.rows[k];
        const [v, lo, hi, hasCi] = vals.slice(i * 4, i * 4 + 4);
        const xv = this.scale(v);
        R.bar.setAttribute('d', hBarPath(x0, R.cy - barH / 2, xv - x0, barH, 4));
        const show = hasCi > 0.5;
        R.ci.style.display = show ? '' : 'none';
        const xl = this.scale(lo), xh = this.scale(hi);
        R.ciLine.setAttribute('x1', xl); R.ciLine.setAttribute('x2', xh);
        R.ciA.setAttribute('x1', xl); R.ciA.setAttribute('x2', xl);
        R.ciB.setAttribute('x1', xh); R.ciB.setAttribute('x2', xh);
        const labelX = Math.max(xv, show ? xh : xv) + 8;
        R.val.setAttribute('x', labelX);
        R.val.textContent = s.fmt(v);
      });
    });
  };

  // ------------------------------------------------------------ grouped columns
  /* spec: {groups:[label], series:[{key,label,cls,hl}], data:{seriesKey:[{v,ci}]}, max, ticks, fmt} */
  function GroupedColumns(container, spec) {
    this.c = container; this.spec = spec;
    this.render();
    observeWidth(container, () => this.render(true));
  }
  GroupedColumns.prototype.render = function (instant) {
    const s = this.spec, c = this.c;
    c.replaceChildren();
    const W = Math.max(280, c.clientWidth || 520);
    const svg = el('svg', { role: 'img', 'aria-label': s.aria || '' }, c);
    const lh = legend(svg, s.series.map((se) => ({ label: se.label, cls: se.cls })), 38, 12, W - 44);
    const left = 38, right = 6, top = lh + 30, bottom = 34, H = top + 206 + bottom;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const ph = H - top - bottom;
    const y = (v) => top + ph * (1 - v / s.max);
    s.ticks.forEach((t) => {
      el('line', { x1: left, x2: W - right, y1: y(t), y2: y(t), class: t === 0 ? 'c-axis' : 'c-grid' }, svg);
      text(svg, left - 8, y(t), s.tickFmt(t), 'c-tick', 'end');
    });
    const gw = (W - left - right) / s.groups.length;
    const n = s.series.length;
    const bw = Math.min(24, (gw - 22) / n - 2);
    const bars = [];
    s.groups.forEach((gLabel, gi) => {
      const gx = left + gi * gw + gw / 2;
      text(svg, gx, H - bottom + 18, gLabel, 'c-label', 'middle');
      s.series.forEach((se, si) => {
        const d = s.data[se.key][gi];
        const x = gx - (n * bw + (n - 1) * 2) / 2 + si * (bw + 2);
        const g = el('g', { class: 'c-row', tabindex: 0 }, svg);
        el('rect', { x: x - 1, y: top, width: bw + 2, height: ph, class: 'c-hit' }, g);
        const p = el('path', { class: se.cls, d: vBarPath(x, y(0), bw, 0, 4) }, g);
        const cx = x + bw / 2;
        let ciG = null;
        if (d.ci) {
          ciG = el('g', { opacity: 0 }, g);
          el('line', { x1: cx, x2: cx, y1: y(d.ci[0]), y2: y(d.ci[1]), class: 'c-ci' }, ciG);
          el('line', { x1: cx - 4, x2: cx + 4, y1: y(d.ci[0]), y2: y(d.ci[0]), class: 'c-ci' }, ciG);
          el('line', { x1: cx - 4, x2: cx + 4, y1: y(d.ci[1]), y2: y(d.ci[1]), class: 'c-ci' }, ciG);
        }
        let lab = null;
        if (se.hl) {
          lab = text(g, cx, (d.ci ? y(d.ci[1]) : y(d.v)) - 10, s.fmt(d.v), 'c-value c-value-hl', 'middle');
          lab.setAttribute('opacity', 0);
        }
        const show = (e) => {
          const lines = [['tt-v', s.fmt(d.v)], ['tt-n', `${se.label} · ${gLabel}`]];
          if (d.ci) lines.push(['tt-s', `95% CI ${s.fmt(d.ci[0])} – ${s.fmt(d.ci[1])}`]);
          if (d.n) lines.push(['tt-s', `n = ${d.n} trials`]);
          tip.show(e, lines, p.getBoundingClientRect());
        };
        g.addEventListener('pointermove', show);
        g.addEventListener('focus', show);
        g.addEventListener('pointerleave', () => tip.hide());
        g.addEventListener('blur', () => tip.hide());
        bars.push({ p, x, bw, v: d.v, ciG, lab });
      });
    });
    const grow = () => tween([0], [1], instant ? 0 : 700, ([k]) => {
      bars.forEach((b) => {
        b.p.setAttribute('d', vBarPath(b.x, y(0), b.bw, (y(0) - y(b.v)) * k, 4));
        if (b.ciG) b.ciG.setAttribute('opacity', k > 0.85 ? 1 : 0);
        if (b.lab) b.lab.setAttribute('opacity', k > 0.85 ? 1 : 0);
      });
    });
    if (instant) grow(); else whenVisible(c, grow);
  };

  // ------------------------------------------------------------ dumbbell
  /* spec: {rows:[{label, a, b}], aLabel, bLabel, max, ticks, fmt} */
  function Dumbbell(container, spec) {
    this.c = container; this.spec = spec;
    this.render();
    observeWidth(container, () => this.render(true));
  }
  Dumbbell.prototype.render = function (instant) {
    const s = this.spec, c = this.c;
    c.replaceChildren();
    const W = Math.max(280, c.clientWidth || 480);
    const narrow = W < 420;
    const svg = el('svg', { role: 'img', 'aria-label': s.aria || '' }, c);
    const lh = legend(svg, [{ kind: 'dot', cls: 'c-bar-2', label: s.aLabel }, { kind: 'dot', cls: 'c-bar-hl', label: s.bLabel }], 0, 12, W);
    const left = narrow ? 80 : 96, right = 20, top = lh + 36, rowH = 50;
    const H = top + rowH * s.rows.length + 6;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const x = (v) => left + (W - left - right) * v / s.max;
    s.ticks.forEach((t) => {
      el('line', { x1: x(t), x2: x(t), y1: top - 8, y2: H - 4, class: t === 0 ? 'c-axis' : 'c-grid' }, svg);
      text(svg, x(t), top - 16, s.tickFmt(t), 'c-tick', 'middle');
    });
    const items = [];
    s.rows.forEach((r, i) => {
      const cy = top + rowH * i + rowH / 2;
      text(svg, left - 14, cy, r.label, 'c-label', 'end');
      const line = el('line', { x1: x(r.a), x2: x(r.a), y1: cy, y2: cy, stroke: 'currentColor', 'stroke-width': 2, class: 'c-ci' }, svg);
      el('circle', { cx: x(r.a), cy, r: 5.5, class: 'c-bar-2', stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
      const b = el('circle', { cx: x(r.a), cy, r: 6.5, class: 'c-bar-hl', stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
      text(svg, x(r.a), cy + 16, s.fmt(r.a), 'c-value', 'middle');
      const lb = text(svg, x(r.b) + 12, cy, s.fmt(r.b), 'c-value c-value-hl');
      lb.setAttribute('opacity', 0);
      const mult = text(svg, (x(r.a) + x(r.b)) / 2, cy - 12, `×${(r.b / r.a).toFixed(1)}`, 'c-tick', 'middle');
      mult.setAttribute('opacity', 0);
      items.push({ r, line, b, lb, mult });
    });
    const go = () => tween([0], [1], instant ? 0 : 900, ([k]) => {
      items.forEach(({ r, line, b, lb, mult }) => {
        const xv = x(r.a + (r.b - r.a) * k);
        line.setAttribute('x2', xv); b.setAttribute('cx', xv);
        lb.setAttribute('opacity', k > 0.9 ? 1 : 0);
        mult.setAttribute('opacity', k > 0.9 ? 1 : 0);
      });
    });
    if (instant) go(); else whenVisible(c, go);
  };

  // ------------------------------------------------------------ frame budget
  /* spec: {budget, segments:[{label, v, cls}], total} */
  function Budget(container, spec) {
    this.c = container; this.spec = spec;
    this.render();
    observeWidth(container, () => this.render(true));
  }
  Budget.prototype.render = function (instant) {
    const s = this.spec, c = this.c;
    c.replaceChildren();
    const W = Math.max(280, c.clientWidth || 480);
    const H = 150, left = 0, right = 8, barY = 58, barH = 26;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': s.aria || '' }, c);
    const max = 36;
    const x = (v) => left + (W - left - right) * v / max;
    // track (the budget)
    el('rect', { x: x(0), y: barY, width: x(s.budget) - x(0), height: barH, rx: 6, fill: 'var(--surface-2)', stroke: 'var(--line-2)', 'stroke-dasharray': '0' }, svg);
    const segs = [];
    let acc = 0;
    s.segments.forEach((sg, i) => {
      const x0 = x(acc) + (i ? 1 : 0), x1 = x(acc + sg.v) - 1;
      const r = el('rect', { x: x0, y: barY, width: 0, height: barH, rx: i === 0 ? 6 : 0, class: sg.cls }, svg);
      // label above
      const lab = text(svg, x0, barY - 26, sg.label, 'c-label');
      const val = text(svg, x0, barY - 10, `${sg.v} ms`, 'c-value c-value-hl');
      lab.setAttribute('opacity', 0); val.setAttribute('opacity', 0);
      segs.push({ r, x0, w: x1 - x0, lab, val });
      acc += sg.v;
    });
    // budget marker
    const bx = x(s.budget);
    el('line', { x1: bx, x2: bx, y1: barY - 6, y2: barY + barH + 6, class: 'c-ci' }, svg);
    text(svg, bx, barY + barH + 20, `${s.budget.toFixed(1)} ms = 30 Hz`, 'c-tick', 'end');
    const tx = x(s.total);
    const tot = text(svg, tx, barY + barH + 38, `${s.total} ms total · ${(s.budget - s.total).toFixed(1)} ms spare`, 'c-value c-value-hl', 'end');
    tot.setAttribute('opacity', 0);
    [0, 10, 20, 30].forEach((t) => text(svg, x(t) + (t === 0 ? 0 : 0), H - 6, t + ' ms', 'c-tick', t === 0 ? 'start' : 'middle'));
    const go = () => tween([0], [1], instant ? 0 : 800, ([k]) => {
      segs.forEach((sg) => {
        sg.r.setAttribute('width', Math.max(0, sg.w * k));
        sg.lab.setAttribute('opacity', k > 0.8 ? 1 : 0);
        sg.val.setAttribute('opacity', k > 0.8 ? 1 : 0);
      });
      tot.setAttribute('opacity', k > 0.95 ? 1 : 0);
    });
    if (instant) go(); else whenVisible(c, go);
  };

  // ------------------------------------------------------------ tables
  function table(container, headers, rows, hlIndex) {
    const t = document.createElement('table');
    const thead = t.createTHead().insertRow();
    headers.forEach((h) => { const th = document.createElement('th'); th.textContent = h; thead.appendChild(th); });
    const tb = t.createTBody();
    rows.forEach((r, i) => {
      const tr = tb.insertRow();
      if (i === hlIndex || (Array.isArray(hlIndex) && hlIndex.includes(i))) tr.className = 'hl';
      r.forEach((cell) => { const td = tr.insertCell(); td.textContent = cell; });
    });
    container.replaceChildren(t);
  }

  function whenVisible(node, cb) {
    if (!('IntersectionObserver' in window)) { cb(); return; }
    const io = new IntersectionObserver((ents) => {
      if (ents.some((e) => e.isIntersecting)) { io.disconnect(); cb(); }
    }, { threshold: 0.3 });
    io.observe(node);
  }

  window.CrossCharts = { HBar, GroupedColumns, Dumbbell, Budget, table, tip, whenVisible, el, text, legend };
})();
