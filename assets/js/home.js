// CROSS site: hub page (highlights playlist, benchmark tabs, video chapters).
(function () {
  'use strict';
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- highlights: a playlist of short clips
  // The clip list (buttons with data-src, data-cap, data-kind, data-d) drives one muted video; the track under it has
  // one segment per clip, as wide as the clip is long: played clips are filled, the one playing fills in the accent.
  var video = document.getElementById('hl-video');
  var chips = Array.prototype.slice.call(document.querySelectorAll('#hl-list .sx-chip'));
  if (video && chips.length) {
    var track = document.getElementById('hl-track');
    var playBtn = document.getElementById('hl-play');
    var fullBtn = document.getElementById('hl-full');
    var cap = document.getElementById('hl-cap');
    var kind = document.getElementById('hl-kind');
    var PAUSE = '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>';
    var PLAY = '<path d="M7 4.5v15l12.5-7.5z"/>';
    var clips = chips.map(function (c) {
      return { src: 'assets/video/' + c.getAttribute('data-src'), cap: c.getAttribute('data-cap'),
        kind: c.getAttribute('data-kind'), d: parseFloat(c.getAttribute('data-d')) || 6 };
    });
    track.style.gridTemplateColumns = clips.map(function (c) { return c.d + 'fr'; }).join(' ');
    var segs = clips.map(function () {
      var s = document.createElement('span'); s.className = 'sx-seg';
      var f = document.createElement('i'); f.className = 'sx-fill';
      s.appendChild(f); track.appendChild(s);
      return s;
    });
    var now = 0, userPaused = reduceMotion, visible = false;

    var paint = function () {
      var p = video.duration ? Math.min(1, video.currentTime / video.duration) : 0;
      segs.forEach(function (s, i) {
        s.style.setProperty('--p', i < now ? 1 : i === now ? p : 0);
        if (i === now) s.setAttribute('data-now', ''); else s.removeAttribute('data-now');
      });
    };
    var setIcon = function () {
      var paused = video.paused;
      playBtn.querySelector('svg').innerHTML = paused ? PLAY : PAUSE;
      playBtn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
    };
    var tryPlay = function () {
      if (userPaused || !visible) return;
      var p = video.play(); if (p && p.catch) p.catch(function () {});
    };
    var load = function (i) {
      now = (i + clips.length) % clips.length;
      var c = clips[now];
      video.poster = c.src + '.jpg';
      video.src = c.src + '.mp4';
      cap.textContent = c.cap;
      kind.textContent = c.kind;
      chips.forEach(function (ch, j) { if (j === now) ch.setAttribute('aria-current', 'true'); else ch.removeAttribute('aria-current'); });
      track.setAttribute('aria-valuenow', String(now + 1));
      track.setAttribute('aria-valuetext', 'Clip ' + (now + 1) + ' of ' + clips.length + ': ' + c.cap);
      paint();
      tryPlay();
    };
    video.addEventListener('timeupdate', paint);
    video.addEventListener('ended', function () { load(now + 1); });
    video.addEventListener('play', setIcon);
    video.addEventListener('pause', setIcon);
    // a smoother fill than timeupdate alone while playing
    var raf = function () { if (!video.paused) paint(); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);

    chips.forEach(function (ch, i) {
      ch.addEventListener('click', function () { userPaused = false; load(i); });
    });
    playBtn.addEventListener('click', function () {
      if (video.paused) { userPaused = false; visible = true; tryPlay(); }
      else { userPaused = true; video.pause(); }
    });
    fullBtn.addEventListener('click', function () {
      var el = video.parentNode;
      if (document.fullscreenElement) document.exitFullscreen();
      else if (el.requestFullscreen) el.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
    });
    // the track: click a segment to jump to its clip; arrow keys step through the clips
    track.addEventListener('click', function (e) {
      var r = track.getBoundingClientRect(), x = e.clientX - r.left, acc = 0, total = 0;
      clips.forEach(function (c) { total += c.d; });
      for (var i = 0; i < clips.length; i++) {
        acc += clips[i].d / total * r.width;
        if (x <= acc) { userPaused = false; load(i); return; }
      }
    });
    track.addEventListener('keydown', function (e) {
      var j = { ArrowRight: now + 1, ArrowLeft: now - 1, Home: 0, End: clips.length - 1 }[e.key];
      if (j !== undefined) { e.preventDefault(); load(j); }
    });
    // play only while on screen
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible) tryPlay(); else if (!video.paused) video.pause();
      }, { threshold: 0.3 }).observe(video);
    } else {
      visible = true;
    }
    chips[0].setAttribute('aria-current', 'true');
    paint(); setIcon(); tryPlay();
  }

  // ---------------------------------------------------------------- benchmark: task tabs (all panels show without JS)
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.bench-tabs [role="tab"]'));
  function select(i, focus) {
    tabs.forEach(function (t, j) {
      var on = i === j;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    if (focus) tabs[i].focus();
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { select(i); });
    t.addEventListener('keydown', function (e) {
      var n = tabs.length, j = { ArrowRight: (i + 1) % n, ArrowLeft: (i - 1 + n) % n, Home: 0, End: n - 1 }[e.key];
      if (j !== undefined) { e.preventDefault(); select(j, true); }
    });
  });
  if (tabs.length) select(0);

  // ---------------------------------------------------------------- video: chapters
  var tour = document.getElementById('tour-video');
  if (!tour) return;
  var chapters = Array.prototype.slice.call(document.querySelectorAll('.chapters button'));
  var starts = chapters.map(function (b) { return parseFloat(b.getAttribute('data-t')); });
  function seek(t) {
    var go = function () {
      try { tour.currentTime = t; } catch (e) {}
      var p = tour.play(); if (p && p.catch) p.catch(function () {});
    };
    if (tour.readyState >= 1) go();
    else { tour.preload = 'auto'; tour.addEventListener('loadedmetadata', go, { once: true }); tour.load(); }
  }
  chapters.forEach(function (b) {
    b.addEventListener('click', function () { seek(parseFloat(b.getAttribute('data-t'))); });
  });
  tour.addEventListener('timeupdate', function () {
    var t = tour.currentTime, on = 0;
    starts.forEach(function (s, i) { if (t >= s - 0.05) on = i; });
    chapters.forEach(function (b, i) {
      b.classList.toggle('is-on', i === on);
      if (i === on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    });
  });
})();
