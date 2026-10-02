(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------ math
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }],
      throwOnError: false,
    });
  }

  // ------------------------------------------------------------ theme
  const themeBtn = $('#themeToggle');
  const isDark = () => {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  themeBtn.addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('cross-theme', next); } catch (e) { /* storage unavailable */ }
    document.dispatchEvent(new Event('themechange'));
  });

  // ------------------------------------------------------------ hero video
  const heroVid = $('.hero-video');
  if (heroVid) {
    if (reduceMotion) { heroVid.removeAttribute('autoplay'); heroVid.pause(); }
    else if ('IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => { e.isIntersecting ? heroVid.play().catch(() => {}) : heroVid.pause(); }).observe(heroVid);
    }
  }

  // ------------------------------------------------------------ nav highlight
  const navLinks = $$('.topnav-links a');
  const sections = navLinks.map((a) => $(a.getAttribute('href'))).filter(Boolean);
  if ('IntersectionObserver' in window) {
    const vis = new Map();
    const io = new IntersectionObserver((ents) => {
      ents.forEach((e) => vis.set(e.target.id, e.intersectionRatio));
      let best = null, br = 0;
      sections.forEach((s) => { const r = vis.get(s.id) || 0; if (r > br) { br = r; best = s.id; } });
      navLinks.forEach((a) => a.classList.toggle('active', best && a.getAttribute('href') === '#' + best));
    }, { threshold: [0, 0.15, 0.3, 0.5, 0.7], rootMargin: '-58px 0px 0px 0px' });
    sections.forEach((s) => io.observe(s));
  }

  // ------------------------------------------------------------ main video + chapters
  const video = $('#mainVideo');
  const chapterBtns = $$('#chapters button');
  const chapterTimes = chapterBtns.map((b) => +b.dataset.t);
  function seek(tSec) {
    $('#video').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    const go = () => { video.currentTime = tSec; video.play().catch(() => {}); };
    if (video.readyState >= 1) go(); else { video.addEventListener('loadedmetadata', go, { once: true }); video.load(); }
  }
  chapterBtns.forEach((b) => b.addEventListener('click', () => {
    const go = () => { video.currentTime = +b.dataset.t; video.play().catch(() => {}); };
    if (video.readyState >= 1) go(); else { video.addEventListener('loadedmetadata', go, { once: true }); video.load(); }
  }));
  video.addEventListener('timeupdate', () => {
    let idx = 0;
    chapterTimes.forEach((ct, i) => { if (video.currentTime >= ct - 0.25) idx = i; });
    chapterBtns.forEach((b, i) => b.classList.toggle('on', i === idx));
  });
  $$('[data-seek]').forEach((b) => b.addEventListener('click', () => seek(+b.dataset.seek)));

  // ------------------------------------------------------------ spotlight figures
  function placeSpot(stage, spot, rect) {
    const img = $('img', stage);
    const sr = stage.getBoundingClientRect(), ir = img.getBoundingClientRect();
    const ox = ir.left - sr.left, oy = ir.top - sr.top;
    const pad = 4;
    spot.style.left = (ox + rect[0] * ir.width - pad) + 'px';
    spot.style.top = (oy + rect[1] * ir.height - pad) + 'px';
    spot.style.width = (rect[2] * ir.width + 2 * pad) + 'px';
    spot.style.height = (rect[3] * ir.height + 2 * pad) + 'px';
  }
  // teaser
  const teaser = $('#teaser');
  if (teaser) {
    const stage = $('.spot-stage', teaser), spot = $('.spot', teaser);
    let pinned = null;
    const keys = $$('.spot-keys li', teaser);
    const showKey = (li) => {
      const b = $('button', li);
      placeSpot(stage, spot, b.dataset.spot.split(',').map(Number));
      spot.classList.add('on');
      keys.forEach((k) => $('button', k).classList.toggle('on', k === li));
    };
    const hide = () => {
      if (pinned) { showKey(pinned); return; }
      spot.classList.remove('on');
      keys.forEach((k) => $('button', k).classList.remove('on'));
    };
    keys.forEach((li) => {
      const b = $('button', li);
      li.addEventListener('pointerenter', () => showKey(li));
      li.addEventListener('pointerleave', hide);
      b.addEventListener('focus', () => showKey(li));
      b.addEventListener('blur', hide);
      b.addEventListener('click', () => { pinned = pinned === li ? null : li; pinned ? showKey(li) : hide(); });
    });
    window.addEventListener('resize', () => { const on = keys.find((k) => $('button', k).classList.contains('on')); if (on) showKey(on); });
  }

  // method steps
  const METHOD_SPOTS = [
    [0.103, 0.125, 0.135, 0.16],   // place recognition
    [0.248, 0.125, 0.313, 0.16],   // relative pose + multi-frame marginalization
    [0.008, 0.305, 0.552, 0.64],   // motion, belief update, hypothesis management
    [0.607, 0.012, 0.384, 0.975],  // map management
  ];
  const mFig = $('.spot-figure-method');
  if (mFig) {
    const stage = $('.spot-stage', mFig), spot = $('.spot', mFig);
    const steps = $$('#steps .step');
    const panels = $$('.step-panel');
    let cur = 0;
    const setStep = (i) => {
      cur = i;
      steps.forEach((s, k) => { s.classList.toggle('on', k === i); s.setAttribute('aria-selected', k === i); });
      panels.forEach((p, k) => p.classList.toggle('on', k === i));
      placeSpot(stage, spot, METHOD_SPOTS[i]);
      spot.classList.add('on');
    };
    steps.forEach((s, i) => {
      s.addEventListener('click', () => setStep(i));
      s.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          const n = (i + (e.key === 'ArrowRight' ? 1 : steps.length - 1)) % steps.length;
          steps[n].focus(); setStep(n);
        }
      });
    });
    const img = $('img', stage);
    const init = () => setStep(cur);
    if (img.complete) init(); else img.addEventListener('load', init);
    window.addEventListener('resize', () => placeSpot(stage, spot, METHOD_SPOTS[cur]));
  }

  // ------------------------------------------------------------ appearance viewers
  $$('[data-viewer]').forEach((v) => {
    const stage = $('.viewer-stage', v);
    const btns = $$('.timeline button', v);
    let idx = 0, timer = null, userTook = false;
    btns.forEach((b) => { const im = new Image(); im.src = b.dataset.src; }); // preload
    const show = (i) => {
      idx = i;
      btns.forEach((b, k) => { b.classList.toggle('on', k === i); b.setAttribute('aria-selected', k === i); });
      const base = $('img:not(.fade-top)', stage);
      const top = document.createElement('img');
      top.className = 'fade-top';
      top.src = btns[i].dataset.src; top.alt = btns[i].dataset.alt;
      top.style.opacity = 0;
      stage.appendChild(top);
      requestAnimationFrame(() => requestAnimationFrame(() => { top.style.opacity = 1; }));
      setTimeout(() => { base.src = top.src; base.alt = top.alt; top.remove(); }, reduceMotion ? 0 : 480);
    };
    btns.forEach((b, i) => {
      b.addEventListener('click', () => { userTook = true; clearInterval(timer); show(i); });
      b.addEventListener('pointerenter', () => { if (userTook) show(i); });
    });
    if (!reduceMotion && 'IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => {
        clearInterval(timer);
        if (e.isIntersecting && !userTook) timer = setInterval(() => show((idx + 1) % btns.length), 2200);
      }, { threshold: 0.5 }).observe(v);
    }
  });

  // ------------------------------------------------------------ per-location figure tabs + lightbox
  const perImg = $('#perLocImg');
  if (perImg) {
    const tabs = $$('.zoomable-wrap .tabs button');
    tabs.forEach((b) => b.addEventListener('click', () => {
      tabs.forEach((o) => o.classList.toggle('on', o === b));
      perImg.style.opacity = 0.3;
      const im = new Image();
      im.onload = () => { perImg.src = b.dataset.src; perImg.width = +b.dataset.w; perImg.height = +b.dataset.h; perImg.style.opacity = 1; };
      im.src = b.dataset.src;
    }));
  }
  const lb = $('#lightbox');
  const lbImg = $('img', lb);
  $$('.zoomable').forEach((z) => z.addEventListener('click', () => {
    const im = $('img', z);
    lbImg.src = im.src; lbImg.alt = im.alt;
    lb.hidden = false;
    $('.lightbox-close', lb).focus();
  }));
  const closeLb = () => { lb.hidden = true; };
  lb.addEventListener('click', closeLb);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !lb.hidden) closeLb(); });

  // ------------------------------------------------------------ robustness videos
  const ROB = {
    occlusion: {
      seek: 255,
      src: 'rob-occlusion',
      label: 'Camera occlusion',
      t: 'The camera is covered twice. With no usable retrieval the motion message keeps moving on odometry and its uncertainty grows, while nothing is written to the map. When a clear frame returns, the measurement message overlaps the motion message again and uncertainty collapses.',
      legend: 'Left: belief on the map. Right: camera. Yellow: estimated trajectory. Blue: motion message. Green: measurement message. Ellipse axes show uncertainty.',
    },
    fast: {
      seek: 229,
      src: 'rob-fast',
      label: 'Fast, blurry motion',
      t: 'Severe motion blur makes observations very uncertain and spawns spurious branches. They live on while the blur lasts, but none is committed. When sharp frames return, temporal filtering removes them and one consistent branch is left.',
      legend: 'Left: belief on the map. Right: camera. Yellow: estimated trajectory. Blue: measurement message. Cyan: motion message. Ellipse axes show uncertainty.',
    },
    noise: {
      seek: 270,
      src: 'rob-noise',
      label: 'Corrupted odometry',
      t: 'Noise is injected into the odometry, up to five times the size of each motion step. The odometry alone drifts far off, yet retrievals keep pulling the estimate back, so it stays close to the reference trajectory.',
      legend: 'Yellow: CROSS estimate. Green: reference trajectory from SLAM. Blue: the corrupted odometry CROSS is given. Noise is injected as T·exp(ξ) with its size set by a signal-to-noise ratio.',
    },
  };
  const robVideo = $('#robVideo'), robCap = $('#robCaption'), robLab = $('#robStepLabel');
  const robLegend = $('#robLegend'), robSeek = $('#robSeek');
  let robKey = 'occlusion';
  function robBuild(autoplay) {
    const sc = ROB[robKey];
    robVideo.poster = `static/video/${sc.src}-poster.jpg`;
    robVideo.src = `static/video/${sc.src}.mp4`;
    robVideo.load();
    if (autoplay) robVideo.play().catch(() => {});
    robLab.textContent = sc.label;
    robCap.textContent = sc.t;
    robLegend.textContent = sc.legend;
    robSeek.textContent = `Watch in the full video ▸ ${Math.floor(sc.seek / 60)}:${String(sc.seek % 60).padStart(2, '0')}`;
  }
  robSeek.addEventListener('click', () => seek(ROB[robKey].seek));
  $$('#robTabs button').forEach((b) => b.addEventListener('click', () => {
    $$('#robTabs button').forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-selected', o === b); });
    robKey = b.dataset.rob; robBuild(true);
  }));
  robBuild();

  // ------------------------------------------------------------ charts
  const CC = window.CrossCharts;
  const pct = (v) => (Math.round(v * 10) / 10).toFixed(1) + '%';

  // Relocalization success (Table 1) with 95% Wilson intervals (Appendix)
  const RELOC = [
    { key: 'orb', label: 'ORB-SLAM3', group: 'SLAM systems', cross: [3.2, 0.5, 1.3], self: [0.3, 4.8, 4.0], ci: { cross: [[2.2, 4.7], [0.2, 1.1]], self: [[0.1, 1.0], [3.7, 6.1]] } },
    { key: 'rtab', label: 'RTAB-Map', group: 'SLAM systems', cross: [29.2, 15.8, 19.8], self: [43.8, 27.5, 33.6], ci: { cross: [[26.1, 32.6], [13.9, 18.0]], self: [[40.2, 47.4], [25.1, 30.1]] } },
    { key: 'mast3r', label: 'MASt3R-SLAM', group: 'SLAM systems', cross: [17.3, 1.4, 6.2], self: [22.6, 2.4, 9.4], ci: { cross: [[14.8, 20.2], [0.9, 2.2]], self: [[19.7, 25.7], [1.7, 3.5]] } },
    { key: 'gm', label: 'GM', group: 'Topological methods', cross: [8.0, 6.4, 6.9], self: [48.3, 26.3, 34.5], ci: { cross: [[6.3, 10.3], [5.2, 7.9]], self: [[44.7, 51.9], [23.9, 28.8]] } },
    { key: 'sm', label: 'SM', group: 'Topological methods', cross: [7.3, 15.2, 12.8], self: [47.9, 33.3, 38.7], ci: { cross: [[5.6, 9.3], [13.3, 17.3]], self: [[44.4, 51.5], [30.8, 36.0]] } },
    { key: 'pbu', label: 'PBU', group: 'Topological methods', cross: [8.0, 6.3, 6.8], self: [48.2, 26.2, 34.4], ci: { cross: [[6.3, 10.3], [5.1, 7.8]], self: [[44.6, 51.8], [23.8, 28.7]] } },
    { key: 'abm', label: 'ABM', group: 'Topological methods', cross: [23.1, 2.0, 9.0], self: [56.9, 22.9, 35.6], ci: { cross: [[20.2, 26.3], [1.4, 3.0]], self: [[53.3, 60.4], [20.6, 25.3]] } },
    { key: 'cross', label: 'CROSS (ours)', group: 'Topological methods', hl: true, cross: [35.3, 39.7, 38.4], self: [47.8, 49.4, 48.8], ci: { cross: [[32.0, 38.8], [37.0, 42.5]], self: [[44.2, 51.4], [46.6, 52.2]] } },
  ];
  const DS = { ols: 0, rover: 1, all: 2 };
  const DS_N = { ols: 735, rover: 1232 };
  const DS_NAME = { ols: 'OpenLORIS', rover: 'Rover', all: 'both datasets' };
  const relocState = { protocol: 'cross', dataset: 'all' };
  const relocValues = () => {
    const o = {};
    RELOC.forEach((m) => {
      const i = DS[relocState.dataset];
      o[m.key] = { v: m[relocState.protocol][i], ci: i < 2 ? m.ci[relocState.protocol][i] : null, n: DS_N[relocState.dataset] };
    });
    return o;
  };
  const reloc = new CC.HBar($('#relocChart'), {
    rows: RELOC.map((m) => ({ key: m.key, label: m.label, group: m.group, hl: m.hl })),
    values: relocValues(), max: 60, ticks: [0, 10, 20, 30, 40, 50, 60],
    fmt: pct, tickFmt: (v) => v + '%', aria: 'Relocalization success rate by method',
  });
  function relocRefresh(dur) {
    const vals = relocValues();
    reloc.update(vals, dur);
    const others = RELOC.filter((m) => !m.hl).map((m) => ({ m, d: vals[m.key] })).sort((a, b) => b.d.v - a.d.v);
    const best = others[0], me = vals.cross;
    const setting = relocState.protocol === 'cross' ? 'with a map from another session' : 'in the same session';
    let msg;
    if (me.v >= best.d.v) {
      msg = `On ${DS_NAME[relocState.dataset]}, ${setting}: CROSS ${pct(me.v)} vs ${pct(best.d.v)} for the next best (${best.m.label}).`;
      if (me.ci && best.d.ci && best.d.ci[1] >= me.ci[0]) msg += ' The intervals overlap, so read this as a higher average rather than a clear separation.';
    } else {
      msg = `On ${DS_NAME[relocState.dataset]}, ${setting}: ${best.m.label} is ahead here (${pct(best.d.v)} vs ${pct(me.v)} for CROSS).`;
    }
    $('#relocFoot').textContent = msg;
    const i = DS[relocState.dataset];
    CC.table($('#relocTable'), ['Method', 'Success', '95% CI'],
      RELOC.map((m) => [m.label, pct(m[relocState.protocol][i]), i < 2 ? `${m.ci[relocState.protocol][i][0]}–${m.ci[relocState.protocol][i][1]}%` : '—']),
      RELOC.length - 1);
  }
  $$('#relocCard [data-filter] button').forEach((b) => b.addEventListener('click', () => {
    const grp = b.parentElement;
    $$('button', grp).forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-checked', o === b); });
    relocState[grp.dataset.filter] = b.dataset.v;
    relocRefresh();
  }));
  reloc.update(relocValues(), 0);
  relocRefresh(0);
  reloc.update(Object.fromEntries(Object.keys(relocValues()).map((k) => [k, { v: 0 }])), 0);
  CC.whenVisible($('#relocChart'), () => relocRefresh(700));

  // Real robot (Table 2 + Wilson intervals)
  const ROBOT = {
    groups: ['LC', 'OR', 'LC+OR', 'All'],
    series: [
      { key: 'orb', label: 'M+S (ORB-SLAM3)', cls: 'c-bar' },
      { key: 'rtab', label: 'M+S (RTAB-Map)', cls: 'c-bar-2' },
      { key: 'cross', label: 'CROSS', cls: 'c-bar-hl', hl: true },
    ],
    data: {
      orb: [{ v: 30, ci: [10.8, 60.3], n: 10 }, { v: 30, ci: [10.8, 60.3], n: 10 }, { v: 20, ci: [5.7, 51.0], n: 10 }, { v: 26.7, ci: [14.2, 44.4], n: 30 }],
      rtab: [{ v: 40, ci: [16.8, 68.7], n: 10 }, { v: 60, ci: [31.3, 83.2], n: 10 }, { v: 30, ci: [10.8, 60.3], n: 10 }, { v: 43.3, ci: [27.4, 60.8], n: 30 }],
      cross: [{ v: 70, ci: [39.7, 89.2], n: 10 }, { v: 80, ci: [49.0, 94.3], n: 10 }, { v: 80, ci: [49.0, 94.3], n: 10 }, { v: 76.7, ci: [59.1, 88.2], n: 30 }],
    },
  };
  new CC.GroupedColumns($('#robotChart'), {
    groups: ROBOT.groups, series: ROBOT.series, data: ROBOT.data,
    max: 100, ticks: [0, 25, 50, 75, 100], tickFmt: (v) => v + '%',
    fmt: (v) => (v % 1 ? v.toFixed(1) : v.toFixed(0)) + '%', aria: 'Object-goal navigation success by setting',
  });
  CC.table($('#robotTable'), ['Method', 'LC', 'OR', 'LC+OR', 'All'],
    ROBOT.series.map((s) => [s.label, ...ROBOT.data[s.key].map((d) => `${d.v}% [${d.ci[0]}, ${d.ci[1]}]`)]), 2);

  // Topo-Bench (Appendix table)
  const TOPO = [
    ['rtab', 'RTAB-Map', 0.059, 0.433, 1, 0.301],
    ['orb', 'ORB-SLAM3', 0.059, 0.183, 1, 0.227],
    ['abm', 'ABM', 0.02, 0.328, 1, 0.201],
    ['ofm', 'OpenFABMAP', 0, 0.112, 0.732, 0.0748],
    ['rat', 'RatSLAM', 0.02, 0.097, 0.443, 0.103],
    ['gm', 'GM', 0.078, 0.302, 0.959, 0.288],
    ['sm', 'SM', 0.118, 0.187, 0.959, 0.281],
    ['pbu', 'PBU', 0.078, 0.302, 0.959, 0.288],
    ['cross', 'CROSS (ours)', 0.275, 0.336, 0.99, 0.452],
  ];
  const TOPO_IDX = { ap: 2, po: 3, ao: 4, bla: 5 };
  const TOPO_DESC = {
    bla: 'Balanced localization accuracy: the geometric mean of the three regimes, so a method cannot win by excelling at one.',
    ap: 'Ambiguous + positive: revisits where a strong look-alike competes with the true place. The hardest regime.',
    po: 'Positive only: revisits without strong look-alikes.',
    ao: 'Ambiguous only: new places that resemble known ones, where the right answer is “not in the map”.',
  };
  const topoVals = (m) => Object.fromEntries(TOPO.map((r) => [r[0], { v: r[TOPO_IDX[m]] }]));
  const f3 = (v) => v.toFixed(3);
  const topo = new CC.HBar($('#topoChart'), {
    rows: TOPO.map((r) => ({ key: r[0], label: r[1], hl: r[0] === 'cross' })),
    values: topoVals('bla'), max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: f3, tickFmt: (v) => v.toString(),
    aria: 'Topo-Bench accuracy by method',
  });
  $('#topoDesc').textContent = TOPO_DESC.bla;
  topo.update(Object.fromEntries(TOPO.map((r) => [r[0], { v: 0 }])), 0);
  CC.whenVisible($('#topoChart'), () => topo.update(topoVals('bla'), 700));
  $$('[data-filter="topo"] button').forEach((b) => b.addEventListener('click', () => {
    $$('[data-filter="topo"] button').forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-checked', o === b); });
    topo.update(topoVals(b.dataset.v));
    $('#topoDesc').textContent = TOPO_DESC[b.dataset.v];
  }));
  CC.table($('#topoTable'), ['Method', 'A+P', 'P.O.', 'A.O.', 'BLA'], TOPO.map((r) => [r[1], ...r.slice(2).map((v) => String(v))]), TOPO.length - 1);

  // Continuous vs discrete (PBU vs CROSS, cross-session)
  new CC.Dumbbell($('#ablationChart'), {
    rows: [
      { label: 'OpenLORIS', a: 8.0, b: 35.3 },
      { label: 'Rover', a: 6.3, b: 39.7 },
      { label: 'Both', a: 6.8, b: 38.4 },
    ],
    aLabel: 'PBU (discrete node IDs)', bLabel: 'CROSS (continuous SE(3))',
    max: 45, ticks: [0, 10, 20, 30, 40], tickFmt: (v) => v + '%', fmt: pct,
    aria: 'Relocalization success of PBU versus CROSS',
  });

  // Runtime
  new CC.Budget($('#budgetChart'), {
    budget: 33.3, total: 28,
    segments: [
      { label: 'Relative pose (PnP)', v: 16, cls: 'c-bar-hl' },
      { label: 'Retrieval, filtering, map', v: 12, cls: 'c-bar-2' },
    ],
    aria: 'Per-frame time: 16 ms relative pose plus 12 ms for retrieval, filtering and map update, within a 33.3 ms budget',
  });
  const poseTime = new CC.HBar($('#poseTimeChart'), {
    rows: [
      { key: 'pnp', label: 'XFeat + PnP', hl: true },
      { key: 'vggt', label: 'VGGT', alt: true },
    ],
    values: { pnp: { v: 16 }, vggt: { v: 490 } }, max: 500, ticks: [0, 100, 200, 300, 400, 500],
    fmt: (v) => Math.round(v) + ' ms', tickFmt: (v) => v + '', aria: 'Relative pose time: 16 ms with PnP, 490 ms with VGGT',
  });
  poseTime.update({ pnp: { v: 0 }, vggt: { v: 0 } }, 0);
  CC.whenVisible($('#poseTimeChart'), () => poseTime.update({ pnp: { v: 16 }, vggt: { v: 490 } }, 900));

  // ------------------------------------------------------------ bibtex
  const copyBtn = $('#copyBib');
  copyBtn.addEventListener('click', async () => {
    const txt = $('#bib').textContent;
    try { await navigator.clipboard.writeText(txt); }
    catch (e) {
      const r = document.createRange(); r.selectNodeContents($('#bib'));
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
      document.execCommand('copy'); s.removeAllRanges();
    }
    copyBtn.textContent = 'Copied';
    setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1600);
  });
})();
