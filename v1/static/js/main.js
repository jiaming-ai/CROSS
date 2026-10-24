(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const smooth = reduceMotion ? 'auto' : 'smooth';

  // ------------------------------------------------------------ math
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }],
      throwOnError: false,
    });
  }
  // The theme is switched by the site header's toggle (site.js), which fires 'themechange' for the demo canvas.

  // ------------------------------------------------------------ tab lists and radio groups: one tab stop, arrow keys move
  // Each widget's own click handler does the selecting; this moves focus and keeps the tab stop on the selected item.
  $$('[role="tablist"], [role="radiogroup"]').forEach((group) => {
    const tabs = group.getAttribute('role') === 'tablist';
    const items = $$(tabs ? '[role="tab"]' : '[role="radio"]', group);
    const state = tabs ? 'aria-selected' : 'aria-checked';
    const sync = () => items.forEach((b) => { b.tabIndex = b.getAttribute(state) === 'true' ? 0 : -1; });
    sync();
    new MutationObserver(sync).observe(group, { subtree: true, attributeFilter: [state] });
    const moves = tabs ? { ArrowRight: 1, ArrowLeft: -1 } : { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    group.addEventListener('keydown', (e) => {
      const i = items.indexOf(document.activeElement);
      if (i < 0) return;
      let n;
      if (moves[e.key]) n = (i + moves[e.key] + items.length) % items.length;
      else if (e.key === 'Home') n = 0;
      else if (e.key === 'End') n = items.length - 1;
      else return;
      e.preventDefault();
      items[n].focus();
      items[n].click();
    });
  });

  // ------------------------------------------------------------ hero video: plays while in view; the corner button pauses it
  const heroVid = $('.hero-video'), heroBtn = $('#heroPause');
  if (heroVid) {
    let paused = reduceMotion;
    const setBtn = () => {
      heroBtn.classList.toggle('paused', paused);
      heroBtn.setAttribute('aria-label', paused ? 'Play background video' : 'Pause background video');
    };
    if (reduceMotion) { heroVid.removeAttribute('autoplay'); heroVid.pause(); }
    setBtn();
    heroBtn.addEventListener('click', () => {
      paused = !paused;
      if (paused) heroVid.pause(); else heroVid.play().catch(() => {});
      setBtn();
    });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => {
        if (e.isIntersecting && !paused) heroVid.play().catch(() => {}); else heroVid.pause();
      }).observe(heroVid);
    }
  }

  // ------------------------------------------------------------ section nav: mark the section being read, keep its link in view
  const header = $('#site-header'), topnav = $('#topnav'), strip = $('.topnav-links', topnav);
  const navLinks = $$('a', strip);
  const sections = navLinks.map((a) => $(a.getAttribute('href')));
  let activeLink = null, spyQueued = false;
  function spy() {
    spyQueued = false;
    const line = header.offsetHeight + topnav.offsetHeight + window.innerHeight * 0.25;
    let cur = null;
    sections.forEach((s, i) => { if (s.getBoundingClientRect().top <= line) cur = navLinks[i]; });
    if (cur === activeLink) return;
    if (activeLink) { activeLink.classList.remove('active'); activeLink.removeAttribute('aria-current'); }
    activeLink = cur;
    if (!cur) return;
    cur.classList.add('active');
    cur.setAttribute('aria-current', 'true');
    // on phones the links scroll sideways: bring the active one into the strip (the page itself does not move)
    const sr = strip.getBoundingClientRect(), ar = cur.getBoundingClientRect();
    if (ar.left < sr.left || ar.right > sr.right) strip.scrollTo({ left: strip.scrollLeft + ar.left - sr.left - 16, behavior: smooth });
  }
  window.addEventListener('scroll', () => { if (!spyQueued) { spyQueued = true; requestAnimationFrame(spy); } }, { passive: true });
  window.addEventListener('resize', spy);
  spy();

  // ------------------------------------------------------------ main video + chapters
  const video = $('#mainVideo');
  const chapterBtns = $$('#chapters button');
  const chapterTimes = chapterBtns.map((b) => +b.dataset.t);
  const playAt = (tSec) => {
    const go = () => { video.currentTime = tSec; video.play().catch(() => {}); };
    if (video.readyState >= 1) go(); else { video.addEventListener('loadedmetadata', go, { once: true }); video.load(); }
  };
  function seek(tSec) {
    $('#video').scrollIntoView({ behavior: smooth, block: 'start' });
    playAt(tSec);
  }
  chapterBtns.forEach((b) => b.addEventListener('click', () => playAt(+b.dataset.t)));
  let chapter = -1;
  video.addEventListener('timeupdate', () => {
    let idx = 0;
    chapterTimes.forEach((ct, i) => { if (video.currentTime >= ct - 0.25) idx = i; });
    if (idx === chapter) return;
    chapter = idx;
    chapterBtns.forEach((b, i) => {
      b.classList.toggle('on', i === idx);
      if (i === idx) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    });
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
  // teaser: hover or focus a key to show its region, click to pin it
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
      b.addEventListener('click', () => {
        pinned = pinned === li ? null : li;
        keys.forEach((k) => $('button', k).setAttribute('aria-pressed', k === pinned));
        if (pinned) showKey(li); else hide();
      });
    });
    window.addEventListener('resize', () => { const on = keys.find((k) => $('button', k).classList.contains('on')); if (on) showKey(on); });
  }

  // method steps (tabs; arrow keys come from the tab-list handler above)
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
    steps.forEach((s, i) => s.addEventListener('click', () => setStep(i)));
    const img = $('img', stage);
    const init = () => setStep(cur);
    if (img.complete) init(); else img.addEventListener('load', init);
    window.addEventListener('resize', () => placeSpot(stage, spot, METHOD_SPOTS[cur]));
  }

  // ------------------------------------------------------------ appearance viewers: one pass through the captures while in view
  // The pass waits while the pointer or focus is on the viewer, and a click on a capture stops it.
  $$('[data-viewer]').forEach((v) => {
    const stage = $('.viewer-stage', v);
    const btns = $$('.timeline button', v);
    let idx = 0, timer = null, userTook = false, advanced = 0, held = false;
    btns.forEach((b) => { const im = new Image(); im.src = b.dataset.src; }); // preload
    const show = (i) => {
      idx = i;
      btns.forEach((b, k) => { b.classList.toggle('on', k === i); b.setAttribute('aria-checked', k === i); });
      const base = $('img:not(.fade-top)', stage);
      const top = document.createElement('img');
      top.className = 'fade-top';
      top.src = btns[i].dataset.src; top.alt = btns[i].dataset.alt;
      top.style.opacity = 0;
      stage.appendChild(top);
      requestAnimationFrame(() => requestAnimationFrame(() => { top.style.opacity = 1; }));
      setTimeout(() => { base.src = top.src; base.alt = top.alt; top.remove(); }, reduceMotion ? 0 : 480);
    };
    const stop = () => { clearInterval(timer); timer = null; };
    btns.forEach((b, i) => {
      b.addEventListener('click', () => { userTook = true; stop(); show(i); });
      b.addEventListener('pointerenter', () => { if (userTook) show(i); });
    });
    v.addEventListener('pointerenter', () => { held = true; });
    v.addEventListener('pointerleave', () => { held = false; });
    v.addEventListener('focusin', () => { held = true; });
    v.addEventListener('focusout', (e) => { if (!v.contains(e.relatedTarget)) held = false; });
    if (!reduceMotion && 'IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => {
        stop();
        if (!e.isIntersecting || userTook || advanced >= btns.length) return;
        timer = setInterval(() => {
          if (held) return;
          show((idx + 1) % btns.length);
          if (++advanced >= btns.length) stop();  // back on the first capture: the pass is over
        }, 2200);
      }, { threshold: 0.5 }).observe(v);
    }
  });

  // ------------------------------------------------------------ per-location figure: three versions, each with its own alt text and method list
  const perImg = $('#perLocImg');
  if (perImg) {
    const opts = $$('.zoomable-wrap .tabs button'), methods = $('#perLocMethods');
    opts.forEach((b) => b.addEventListener('click', () => {
      opts.forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-checked', o === b); });
      methods.textContent = b.dataset.caption;
      perImg.style.opacity = 0.3;
      const im = new Image();
      im.onload = () => {
        perImg.src = b.dataset.src; perImg.width = +b.dataset.w; perImg.height = +b.dataset.h; perImg.alt = b.dataset.alt;
        perImg.style.opacity = 1;
      };
      im.src = b.dataset.src;
    }));
  }

  // ------------------------------------------------------------ lightbox: a modal <dialog>; a click anywhere closes it
  const lb = $('#lightbox'), lbImg = $('img', lb);
  let lbFrom = null;
  $$('.zoomable').forEach((z) => z.addEventListener('click', () => {
    const im = $('img', z);
    lbImg.src = im.src; lbImg.alt = im.alt;
    lbFrom = z;
    lb.showModal();
  }));
  lb.addEventListener('click', () => lb.close());
  lb.addEventListener('close', () => { if (lbFrom) lbFrom.focus(); });

  // ------------------------------------------------------------ robustness videos (the HTML holds the first scenario)
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
  const robVideo = $('#robVideo'), robPanel = $('#rob'), robCap = $('#robCaption'), robLab = $('#robStepLabel');
  const robLegend = $('#robLegend'), robSeek = $('#robSeek');
  let robKey = 'occlusion';
  function robBuild() {
    const sc = ROB[robKey];
    robVideo.poster = `static/video/${sc.src}-poster.jpg`;
    robVideo.src = `static/video/${sc.src}.mp4`;
    robVideo.load();
    robVideo.play().catch(() => {});
    robLab.textContent = sc.label;
    robCap.textContent = sc.t;
    robLegend.textContent = sc.legend;
    robSeek.textContent = `Watch in the full video ▸ ${Math.floor(sc.seek / 60)}:${String(sc.seek % 60).padStart(2, '0')}`;
  }
  robSeek.addEventListener('click', () => seek(ROB[robKey].seek));
  $$('#robTabs button').forEach((b) => b.addEventListener('click', () => {
    $$('#robTabs button').forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-selected', o === b); });
    robPanel.setAttribute('aria-labelledby', b.id);
    if (robKey === b.dataset.rob) return;
    robKey = b.dataset.rob;
    robBuild();
  }));

  // ------------------------------------------------------------ charts (the HTML holds every table and the default texts)
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
  // one axis for every view (the highest interval end is 60.4%), so switching views compares like with like
  const reloc = new CC.HBar($('#relocChart'), {
    rows: RELOC.map((m) => ({ key: m.key, label: m.label, group: m.group, hl: m.hl })),
    values: relocValues(), max: 70, ticks: [0, 10, 20, 30, 40, 50, 60, 70],
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
  reloc.update(Object.fromEntries(RELOC.map((m) => [m.key, { v: 0 }])), 0);
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
  topo.update(Object.fromEntries(TOPO.map((r) => [r[0], { v: 0 }])), 0);
  CC.whenVisible($('#topoChart'), () => topo.update(topoVals('bla'), 700));
  $$('[data-filter="topo"] button').forEach((b) => b.addEventListener('click', () => {
    $$('[data-filter="topo"] button').forEach((o) => { o.classList.toggle('on', o === b); o.setAttribute('aria-checked', o === b); });
    topo.update(topoVals(b.dataset.v));
    $('#topoDesc').textContent = TOPO_DESC[b.dataset.v];
  }));

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
