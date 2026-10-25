// CROSS site: docs page (the index of the sections with scrollspy, the mode picker, the folder-layout tabs). site.js,
// loaded first, names the heading anchors, makes code and tables that scroll sideways reachable by keyboard
// (CROSS.markScrollers) and puts a linked section under the bars once the fonts are in.
(function () {
  'use strict';
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var smooth = reduceMotion ? 'auto' : 'smooth';
  var CROSS = window.CROSS || {};
  var markScrollers = function () { if (CROSS.markScrollers) CROSS.markScrollers(); };

  // ---------------------------------------------------------------- the index: mark the section in view
  var toc = document.querySelector('.docs-toc');
  var nav = toc && toc.querySelector('nav');
  var bar = toc && toc.querySelector('.docs-toc-scroll');
  var links = toc ? Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]')) : [];
  var heads = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
  var current = -1, pinned = -1, ticking = false;
  // how far below the window's top a linked section lands: its scroll margin (docs.css --docs-top)
  var landing = CROSS.landing || function (el) { return (el && parseFloat(getComputedStyle(el).scrollMarginTop)) || 0; };

  // keep the current link in view: in the bar (narrow screens) or in the sidebar when it is taller than the window.
  // Instantly: while an index scrolls smoothly, Chrome drops the page's jump to a section clicked in the meantime.
  function reveal(a) {
    var r = a.getBoundingClientRect(), box;
    if (bar && bar.scrollWidth > bar.clientWidth) {
      box = bar.getBoundingClientRect();
      if (r.left < box.left + 24 || r.right > box.right - 24) {
        bar.scrollLeft += r.left - box.left - (box.width - r.width) / 2;
      }
    } else if (nav && nav.scrollHeight > nav.clientHeight) {
      box = nav.getBoundingClientRect();
      if (r.top < box.top + 24 || r.bottom > box.bottom - 24) {
        nav.scrollTop += r.top - box.top - (box.height - r.height) / 2;
      }
    }
  }
  function spy() {
    ticking = false;
    // a section is current once its title passes a line a third of the way down the window
    var line = Math.max(landing(heads[0]), window.innerHeight * 0.3) + 8;
    var idx = -1;
    heads.forEach(function (h, i) { if (h && h.getBoundingClientRect().top <= line) idx = i; });
    // at the bottom of the page the last titles cannot reach the line: the clicked one, else the last
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
      idx = pinned >= 0 ? Math.max(pinned, idx) : heads.length - 1;
    }
    if (idx === current) return;
    if (current >= 0) links[current].removeAttribute('aria-current');
    current = idx;
    if (idx >= 0) { links[idx].setAttribute('aria-current', 'true'); reveal(links[idx]); }
    // back above the first section: the index from its start again
    else if (bar && bar.scrollWidth > bar.clientWidth) bar.scrollLeft = 0;
    else if (nav && nav.scrollHeight > nav.clientHeight) nav.scrollTop = 0;
  }
  function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(spy); } }
  if (links.length) {
    toc.addEventListener('click', function (e) { var a = e.target.closest('a'); pinned = a ? links.indexOf(a) : -1; });
    // the reader scrolls on their own again: forget the clicked link
    ['wheel', 'pointerdown', 'keydown'].forEach(function (t) { window.addEventListener(t, function () { pinned = -1; }, { passive: true }); });
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    spy();
  }

  // ---------------------------------------------------------------- mode picker (docs/GUIDE.md "Choose a mode", docs/USAGE.md "Modes")
  // install: the install.sh option the mode needs; text: what it uses; bench: its benchmark system (setup = the prefix
  // of the mode; benchmark/configs/systems.yaml, and the "+vio" rows run the *_basalt mode files, USAGE.md section 1)
  var MODES = {
    stereo_wheel: { install: '--stereo', bench: 'cross_stereo', text: 'Stereo pair + external odometry. VGGT-Omega measures the pose against the retrieved keyframes and the stereo baseline gives the metric scale.' },
    stereo_basalt: { install: '--stereo', bench: 'cross_stereo+vio', basalt: true, text: 'Stereo pair + the odometry of Basalt’s stereo-inertial VIO, read from odom_vio.txt.' },
    stereo_imu: { install: '--stereo', bench: 'cross_stereo_vgio', imu: true, text: 'Stereo pair + IMU: CROSS fuses the IMU with its own VGGT-Omega passes, with the scale from the stereo pair. No separate odometry needed; the pairs must be rectified.' },
    stereo_dpvo: { install: '--mono', bench: 'cross_stereo_vo', text: 'Stereo pair only: CROSS runs DPVO visual odometry, with the scale from stereo depth. The pairs must be rectified.' },
    rgbd_wheel: { install: '', bench: 'cross_rgbd', text: 'Color + depth + external odometry. Feature matching (XFeat + LightGlue) and PnP on the depth: no large model and no non-commercial weights.' },
    rgbd_basalt: { install: '', bench: 'cross_rgbd+vio', basalt: true, text: 'Color + depth + the odometry of a Basalt stereo-inertial VIO, read from odom_vio.txt.' },
    rgbd_dpvo: { install: '--mono', bench: 'cross_rgbd_vo', text: 'Color + depth only: CROSS runs DPVO visual odometry, with the scale from the depth.' },
    rgbd_ff_wheel: { install: '--stereo', bench: 'cross_rgbd_ff', text: 'Color + depth + external odometry, with VGGT-Omega relative poses scaled by the depth.' },
    rgbd_ff_basalt: { install: '--stereo', bench: 'cross_rgbd_ff+vio', basalt: true, text: 'Color + depth, VGGT-Omega scaled by the depth, with the odometry of a Basalt VIO (odom_vio.txt).' },
    mono_wheel: { install: '--stereo', bench: 'cross_mono_ff_odom', text: 'One color camera + external odometry (wheel, or a phone’s ARKit / ARCore). VGGT-Omega, with the metric scale from the odometry and the map.' },
    mono_basalt: { install: '--stereo', bench: 'cross_mono_ff_odom+vio', basalt: true, text: 'One color camera + the odometry of a Basalt stereo-inertial VIO (odom_vio.txt).' },
    mono_imu: { install: '--mono', bench: 'cross_mono_ff_vgio', imu: true, text: 'One color camera + IMU: VGGT-Omega and the IMU in a local pose graph, with the scale from learned depth.' },
    mono_vio: { install: '--mono', bench: 'cross_mono_ff_vio', imu: true, text: 'One color camera + IMU: DPVO scaled by the IMU (a reference system).' },
    mono_dpvo: { install: '--mono', bench: 'cross_mono', text: 'One color camera only: DPVO visual odometry, with the scale from learned depth (Depth Anything 3).' }
  };
  // camera -> motion input -> [recommended, [other options with a short reason]]
  var PICK = {
    stereo: {
      odom: ['stereo_wheel', [['stereo_dpvo', 'which needs no odometry']]],
      imu: ['stereo_basalt', [['stereo_imu', 'where CROSS fuses the IMU itself, with no Basalt file']]],
      none: ['stereo_dpvo', []]
    },
    rgbd: {
      odom: ['rgbd_wheel', [['rgbd_ff_wheel', 'VGGT-Omega scaled by the depth, for large changes in lighting and viewpoint']]],
      imu: ['rgbd_basalt', [['rgbd_ff_basalt', 'VGGT-Omega scaled by the depth'], ['rgbd_dpvo', 'without a stereo pair for Basalt; it ignores the IMU']]],
      none: ['rgbd_dpvo', []]
    },
    mono: {
      odom: ['mono_wheel', []],
      imu: ['mono_imu', [['mono_vio', 'DPVO scaled by the IMU'], ['mono_basalt', 'with the odometry of a Basalt VIO, which needs a stereo pair and the IMU']]],
      none: ['mono_dpvo', []]
    }
  };
  var wrap = document.querySelector('.modes-wrap');
  var picker = document.getElementById('mode-picker');
  if (picker) {
    var $ = function (id) { return document.getElementById(id); };
    var chipsByMode = {};
    document.querySelectorAll('.mode-chip[data-mode]').forEach(function (el) { chipsByMode[el.getAttribute('data-mode')] = el; });
    var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
    var code = function (s) { return '<code>' + esc(s) + '</code>'; };
    var val = function (name) { var el = picker.querySelector('input[name="' + name + '"]:checked'); return el ? el.value : ''; };
    var picked = null;
    var bench = null;  // mode -> {t3, t1}: overall relocalization success and mapping error in the benchmark

    // the picked mode in the table: scroll the table (not the page) when the mode is off-screen
    var showPick = function (chip) {
      if (!wrap || wrap.scrollWidth <= wrap.clientWidth) return;
      var w = wrap.getBoundingClientRect(), c = chip.getBoundingClientRect();
      var left = w.left + chip.closest('tr').querySelector('th').offsetWidth;  // right of the sticky first column
      if (c.left >= left && c.right <= w.right - 8) return;
      wrap.scrollTo({ left: wrap.scrollLeft + c.left - left - (w.right - left - c.width) / 2, behavior: smooth });
    };
    // "(94% relocalization success, 0.69% mapping error)" of a mode, or '' before the numbers have loaded
    var stats = function (mode) {
      var b = bench && bench[mode];
      if (!b || !b.t3) return '';
      return ' (' + b.t3 + ' relocalization success' + (b.t1 ? ', ' + b.t1 + ' mapping error' : '') + ')';
    };

    var render = function (announce) {
      var cam = val('pk-cam'), odom = val('pk-odom'), outdoor = val('pk-env') === 'outdoor';
      var vio = odom === 'vio';  // VIO poses are external odometry: the *_wheel mode with the VIO overlay
      var rec = PICK[cam][vio ? 'odom' : odom], mode = rec[0], m = MODES[mode];
      var install = 'bash install.sh' + (m.install ? ' ' + m.install : '');
      var configs = [];
      if (outdoor) configs.push('configs/overlays/outdoor.yaml');
      if (vio) configs.push('configs/overlays/odometry_vio.yaml');

      $('pk-mode').textContent = mode;
      $('pk-install').textContent = install;
      $('pk-desc').textContent = m.text;
      var b = bench && bench[mode];
      $('pk-bench').hidden = !(b && b.t3);
      if (b && b.t3) {
        $('pk-bench').innerHTML = 'In the benchmark: <b>' + b.t3 + '</b> relocalization success' +
          (b.t1 ? ', <b>' + b.t1 + '</b> mapping error (of the path)' : '') + '. <a href="../leaderboard/">Benchmark and caveats</a>';
      }
      // one line per command line, so that phones wrap them with a hanging indent
      var lines = ['<span class="p">$ </span>python run.py my_sequence --mode ' + esc(mode) + (configs.length ? ' \\' : '')];
      if (configs.length) lines.push('    --config ' + esc(configs.join(' ')));
      $('pk-cmd').innerHTML = lines.map(function (l) { return '<span class="ln">' + l + '</span>'; }).join('');

      var notes = [];
      if (vio) notes.push('Poses from a low-drift VIO (a phone’s ARKit / ARCore, or a Basalt-class VIO on the robot) go in ' + code('odom_left.txt') + '; ' + code('odometry_vio.yaml') + ' sets how little they drift.');
      if (m.basalt) notes.push(code('odom_vio.txt') + ' holds one pose per frame (as ' + code('odom_left.txt') + ') from Basalt’s stereo-inertial VIO, which needs a stereo pair and an IMU. ' + code('scripts/vio/install_basalt.sh') + ' builds Basalt; ' + code('benchmark/datasets/prepare_vio.py') + ' writes the file for the benchmark datasets.');
      if (m.imu) notes.push('The folder needs ' + code('imu.txt') + ' and ' + code('imu.json') + ' (<a href="#file-formats">file formats</a>).');
      if (outdoor) notes.push(code('outdoor.yaml') + ' sets the outdoor scale (larger distances).');
      if (!vio) rec[1].forEach(function (alt) { notes.push('Also possible: ' + code(alt[0]) + stats(alt[0]) + ', ' + esc(alt[1]) + '.'); });
      $('pk-notes').innerHTML = notes.map(function (n) { return '<li>' + n + '</li>'; }).join('');
      $('pk-notes').hidden = !notes.length;

      if (picked) picked.classList.remove('is-pick');
      picked = chipsByMode[mode] || null;
      if (picked) { picked.classList.add('is-pick'); showPick(picked); }
      // a short summary for screen readers (the output above is not a live region: it is long)
      if (announce) $('pk-status').textContent = 'Recommended: ' + mode + ', installed with ' + install + '.';
    };
    picker.addEventListener('change', function () { render(true); markScrollers(); });
    // the HTML already shows the answer to the default questions, with its benchmark numbers (tools/build.py), so the
    // page does not move while the numbers load; a form the browser restored with other answers is drawn now
    if (val('pk-cam') === 'stereo' && val('pk-odom') === 'odom' && val('pk-env') === 'indoor') {
      picked = chipsByMode.stereo_wheel || null;
      if (picked) { picked.classList.add('is-pick'); showPick(picked); }
    } else {
      render(false);
    }

    // the benchmark numbers of every mode (overall T3 at the larger radius, overall T1; tools/build.py writes the file)
    fetch('../assets/data/leaderboard.json').then(function (r) { return r.ok ? r.json() : null; }).then(function (L) {
      if (!L) return;
      var overall = function (tid, system, setup) {
        var row = L.tasks[tid].rows.filter(function (r) { return r.system === system && r.setup === setup; })[0];
        var c = row && row.cells.overall;
        // T1's mean leaves failed sequences out, so it is not shown then (T3 counts their trials as failed)
        return c && c.v != null && !(tid === 't1' && c.fail) ? c.v : null;
      };
      bench = {};
      Object.keys(MODES).forEach(function (mode) {
        var setup = mode.split('_')[0], t3 = overall('t3', MODES[mode].bench, setup), t1 = overall('t1', MODES[mode].bench, setup);
        bench[mode] = { t3: t3 && Math.round(t3[1] * 100) + '%', t1: t1 != null ? t1.toFixed(2) + '%' : null };
      });
      render(false);
    }).catch(function () {});
  }

  // ---------------------------------------------------------------- tabs (folder layouts): arrow keys move between tabs
  document.querySelectorAll('[data-tabs]').forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    var select = function (i, focus) {
      tabs.forEach(function (t, j) {
        var on = i === j;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
      });
      if (focus) tabs[i].focus();
      markScrollers();  // the panel shown now may scroll sideways
    };
    box.setAttribute('data-ready', '');
    tabs.forEach(function (t, i) {
      t.addEventListener('click', function () { select(i); });
      t.addEventListener('keydown', function (e) {
        var n = tabs.length, j = { ArrowRight: (i + 1) % n, ArrowLeft: (i - 1 + n) % n, Home: 0, End: n - 1 }[e.key];
        if (j !== undefined) { e.preventDefault(); select(j, true); }
      });
    });
    select(0);
  });
})();
