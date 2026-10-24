// CROSS site: shared behaviour (theme, header, mobile menu, copy buttons, reveal on scroll, in-view video playback).
(function () {
  'use strict';
  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- theme (shared key with the CROSS 1 page)
  function effectiveDark() {
    var t = root.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  document.querySelectorAll('[data-theme-toggle]').forEach(function (b) {
    b.addEventListener('click', function () {
      var next = effectiveDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('cross-theme', next); } catch (e) {}
      document.dispatchEvent(new CustomEvent('cross:theme', { detail: next }));
      document.dispatchEvent(new Event('themechange'));  // the CROSS 1 page re-draws its charts on this
    });
  });

  // ---------------------------------------------------------------- header: solid once the page scrolls (dark hero pages)
  var header = document.getElementById('site-header');
  if (header && document.body.classList.contains('night-top')) {
    var onScroll = function () { header.classList.toggle('is-solid', window.scrollY > 24); };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // ---------------------------------------------------------------- mobile menu
  var menuBtn = document.querySelector('.sh-menu');
  var drawer = document.getElementById('sh-drawer');
  if (menuBtn && drawer) {
    var setOpen = function (open) {
      menuBtn.setAttribute('aria-expanded', String(open));
      menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
      drawer.hidden = !open;
      if (header) header.classList.toggle('is-open', open);
    };
    menuBtn.addEventListener('click', function () { setOpen(menuBtn.getAttribute('aria-expanded') !== 'true'); });
    drawer.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  }

  // ---------------------------------------------------------------- copy buttons: <button class="copy" data-copy="#id | text">
  var COPY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';
  document.querySelectorAll('.copy').forEach(function (btn) {
    if (!btn.innerHTML.trim()) btn.innerHTML = COPY_ICON + '<span>Copy</span>';
    btn.addEventListener('click', function () {
      var src = btn.getAttribute('data-copy');
      var text;
      if (src && src.charAt(0) === '#') {
        var el = document.querySelector(src);
        text = el ? el.innerText : '';
      } else if (src) {
        text = src;
      } else {
        var pre = btn.closest('.code').querySelector('pre');
        text = pre ? pre.innerText : '';
      }
      // drop shell prompts and comment-only lines so the copied block runs as is
      if (btn.hasAttribute('data-strip-prompt')) {
        text = text.split('\n').filter(function (l) { return !/^\s*#/.test(l); })
          .map(function (l) { return l.replace(/^\s*\$ /, ''); }).join('\n').trim();
      }
      var done = function () {
        btn.classList.add('is-done');
        var label = btn.querySelector('span');
        if (label) label.textContent = 'Copied';
        setTimeout(function () { btn.classList.remove('is-done'); if (label) label.textContent = 'Copy'; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        ta.remove();
      }
    });
  });

  // ---------------------------------------------------------------- reveal on scroll
  var revealEls = document.querySelectorAll('[data-reveal]');
  if ('IntersectionObserver' in window && !reduceMotion) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('is-in'); io.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('is-in'); });
  }

  // ---------------------------------------------------------------- muted loops: play only while visible
  // <video data-inview muted loop playsinline preload="none" poster="..."> ; with reduced motion they stay paused
  var loops = document.querySelectorAll('video[data-inview]');
  if (loops.length && 'IntersectionObserver' in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting && !reduceMotion) {
          if (v.preload === 'none') v.preload = 'auto';
          var p = v.play(); if (p && p.catch) p.catch(function () {});
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { threshold: 0.35 });
    loops.forEach(function (v) { v.muted = true; vio.observe(v); });
  }

  // ---------------------------------------------------------------- headings in long pages get a copyable anchor
  document.querySelectorAll('[data-anchors] h2[id], [data-anchors] h3[id]').forEach(function (h) {
    var a = document.createElement('a');
    a.className = 'h-anchor'; a.href = '#' + h.id; a.setAttribute('aria-label', 'Link to this section');
    a.textContent = '#';
    h.appendChild(a);
  });
})();
