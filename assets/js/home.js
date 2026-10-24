// CROSS site: hub page (benchmark tabs, video chapters, "watch in the video" links).
(function () {
  'use strict';

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

  // ---------------------------------------------------------------- video tour: chapters
  var video = document.getElementById('tour-video');
  if (!video) return;
  var chapters = Array.prototype.slice.call(document.querySelectorAll('.chapters button'));
  var starts = chapters.map(function (b) { return parseFloat(b.getAttribute('data-t')); });

  function seek(t, play) {
    var go = function () {
      try { video.currentTime = t; } catch (e) {}
      if (play) { var p = video.play(); if (p && p.catch) p.catch(function () {}); }
    };
    if (video.readyState >= 1) go();
    else { video.preload = 'auto'; video.addEventListener('loadedmetadata', go, { once: true }); video.load(); }
  }
  chapters.forEach(function (b) {
    b.addEventListener('click', function () { seek(parseFloat(b.getAttribute('data-t')), true); });
  });
  video.addEventListener('timeupdate', function () {
    var t = video.currentTime, on = 0;
    starts.forEach(function (s, i) { if (t >= s - 0.05) on = i; });
    chapters.forEach(function (b, i) {
      b.classList.toggle('is-on', i === on);
      if (i === on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    });
  });

  // ---------------------------------------------------------------- "watch in the video" buttons and the hero link
  var tour = document.getElementById('tour');
  function toTour(t) {
    tour.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    seek(t, true);
  }
  document.querySelectorAll('[data-seek]').forEach(function (b) {
    b.addEventListener('click', function () { toTour(parseFloat(b.getAttribute('data-seek'))); });
  });
  document.querySelectorAll('[data-play-tour]').forEach(function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); toTour(0); });
  });
})();
