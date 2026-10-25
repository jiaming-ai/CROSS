// CROSS site: shared behavior (theme, header, mobile menu, copy buttons, reveal on scroll, in-view video playback;
// on the pages built on site.css also heading anchors, keyboard access to boxes that scroll sideways, and links to a
// section that land under the bar once the web fonts are in). Page scripts use window.CROSS (below).
(function () {
  'use strict';
  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // shared with the page scripts, which load after this one
  var CROSS = window.CROSS = window.CROSS || {};
  // the pages built on site.css; the NeurIPS paper page (v1/) loads only chrome.css and has its own main.js for the
  // scrolling boxes and the links to a section
  var sitePage = !!document.querySelector('link[rel="stylesheet"][href$="assets/css/site.css"]');
  // a heading's words without its number and its anchor ("Install", not "02 Install #")
  var headingText = CROSS.headingText = function (h) {
    var c = h.cloneNode(true);
    c.querySelectorAll('.sx-n, .h-anchor').forEach(function (el) { el.remove(); });
    return c.textContent.replace(/\s+/g, ' ').trim();
  };
  // how far below the window's top a linked element lands: its scroll margin (site.css), plus the page's scroll
  // padding where a page still sets one
  var landing = CROSS.landing = function (el) {
    var pad = parseFloat(getComputedStyle(root).scrollPaddingTop) || 0;
    return pad + ((el && parseFloat(getComputedStyle(el).scrollMarginTop)) || 0);
  };
  // a reload, or a visit through back / forward that the page cache did not keep: the browser puts the reader back
  // where they were, so the page scripts leave the scroll position and the focus alone
  CROSS.restored = (function () {
    try {
      var nav = performance.getEntriesByType('navigation')[0];
      var type = nav ? nav.type : { 1: 'reload', 2: 'back_forward' }[performance.navigation && performance.navigation.type];
      return type === 'reload' || type === 'back_forward';
    } catch (e) { return false; }
  })();

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
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape' || menuBtn.getAttribute('aria-expanded') !== 'true') return;
      var inside = drawer.contains(document.activeElement);
      setOpen(false);
      if (inside) menuBtn.focus();  // keep the reader's place
    });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  }

  // ---------------------------------------------------------------- copy buttons: <button class="copy" data-copy="#id | text">
  var live = document.createElement('p');
  live.className = 'sr-only'; live.setAttribute('aria-live', 'polite');
  document.body.appendChild(live);
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
        live.textContent = '';
        setTimeout(function () { live.textContent = 'Copied to clipboard'; }, 50);
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
  // <video data-inview muted loop playsinline preload="none" poster="...">: plays while on screen (never with reduced
  // motion) and gets a pause / play button in its parent (which must be position: relative); a loop the reader paused
  // stays paused. data-no-toggle skips the button.
  var PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>';
  var PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>';
  var play = function (v) {
    if (v.preload === 'none') v.preload = 'auto';
    var p = v.play(); if (p && p.catch) p.catch(function () {});
  };
  // posters of videos far down the page load only when the reader gets near them: <video data-poster="...">
  var lazy = document.querySelectorAll('video[data-poster]');
  if (lazy.length && 'IntersectionObserver' in window) {
    var pio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        en.target.poster = en.target.getAttribute('data-poster');
        pio.unobserve(en.target);
      });
    }, { rootMargin: '800px 0px' });
    lazy.forEach(function (v) { pio.observe(v); });
  } else {
    lazy.forEach(function (v) { v.poster = v.getAttribute('data-poster'); });
  }
  var loops = document.querySelectorAll('video[data-inview]');
  loops.forEach(function (v) {
    v.muted = true;
    if (v.hasAttribute('data-no-toggle')) return;
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'media-toggle';
    var sync = function () {
      btn.innerHTML = v.paused ? PLAY : PAUSE;
      btn.setAttribute('aria-label', v.paused ? 'Play video' : 'Pause video');
    };
    btn.addEventListener('click', function () {
      if (v.paused) { delete v.dataset.userPaused; play(v); }
      else { v.dataset.userPaused = '1'; v.pause(); }
    });
    v.addEventListener('play', sync);
    v.addEventListener('pause', sync);
    v.parentNode.appendChild(btn);
    sync();
  });
  if (loops.length && 'IntersectionObserver' in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting && !reduceMotion && !v.dataset.userPaused) play(v);
        else if (!v.paused) v.pause();
      });
    }, { threshold: 0.35 });
    loops.forEach(function (v) { vio.observe(v); });
  }

  // ---------------------------------------------------------------- headings in long pages get a copyable anchor
  // named after its heading ("Link to the section “Install”"); site.css hides it on touch screens. The heading is named
  // with its own words (aria-label, without what it hides from screen readers, such as the section number): a labelled
  // link inside a heading adds its label to the heading's name, so a screen reader's list of headings, and the section
  // the heading labels, would read "Install Link to the section “Install”".
  document.querySelectorAll('[data-anchors] h2[id], [data-anchors] h3[id]').forEach(function (h) {
    if (!h.hasAttribute('aria-label') && !h.hasAttribute('aria-labelledby')) {
      var c = h.cloneNode(true);
      c.querySelectorAll('.h-anchor, [aria-hidden="true"]').forEach(function (el) { el.remove(); });
      h.setAttribute('aria-label', c.textContent.replace(/\s+/g, ' ').trim());
    }
    var a = document.createElement('a');
    a.className = 'h-anchor'; a.href = '#' + h.id;
    a.setAttribute('aria-label', 'Link to the section “' + headingText(h) + '”');
    a.textContent = '#';
    h.appendChild(a);
  });

  if (!sitePage) return;

  // ---------------------------------------------------------------- code, listings and tables that scroll sideways:
  // reachable by keyboard (Safari does not focus scrolling boxes by itself) while they overflow, named after their
  // section and label. A listing (an ol) keeps its list role and gets only the stop and the name; the rest also get
  // role="region". CROSS.markScrollers() checks again (page scripts call it after they change a box's content or show
  // a hidden one).
  var scrollLabel = function (el) {
    // a quick-start step is named after the step, everything else after its section
    var step = el.closest('.qs-steps > li'), sec = el.closest('section');
    var h = (step && step.querySelector('h3')) || (sec && sec.querySelector('h2'));
    var cap = el.querySelector('caption');
    var head = el.closest('.code') && el.closest('.code').querySelector('.code-head span');
    var group = el.previousElementSibling;  // a listing's group head: <p class="sx-list-head"><span>Terminal</span>
    group = group && group.matches('.sx-list-head') && group.querySelector('span');
    var what = (cap ? cap.textContent : head ? head.textContent : group ? group.textContent : el.closest('.sx-bib') ? 'BibTeX'
      : el.matches('.table-wrap') ? 'Table' : 'Code').trim();
    var where = h ? headingText(h) : '';
    return (where && where.toLowerCase() !== what.toLowerCase() ? where + ': ' : '') + what;
  };
  var markScrollers = CROSS.markScrollers = function () {
    document.querySelectorAll('pre, .table-wrap, .sx-listing').forEach(function (el) {
      var ox = getComputedStyle(el).overflowX;
      var scrolls = (ox === 'auto' || ox === 'scroll') && el.scrollWidth > el.clientWidth + 1;
      if (scrolls && !el.hasAttribute('tabindex')) {
        var list = el.matches('ol, ul');
        el.setAttribute('tabindex', '0');
        if (!list) el.setAttribute('role', 'region');
        el.setAttribute('aria-label', scrollLabel(el) + ' (scrolls sideways)');
        el.dataset.scrollMarked = list ? 'list' : 'region';
      } else if (!scrolls && el.dataset.scrollMarked) {
        el.removeAttribute('tabindex');
        el.removeAttribute('aria-label');
        if (el.dataset.scrollMarked === 'region') el.removeAttribute('role');
        delete el.dataset.scrollMarked;
      }
    });
  };
  var scrollTimer = 0;
  window.addEventListener('resize', function () { clearTimeout(scrollTimer); scrollTimer = setTimeout(markScrollers, 150); });
  // after the fonts arrive (they change the widths), and once now
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(markScrollers);
  markScrollers();

  // ---------------------------------------------------------------- a link to a section (docs/#install): the browser
  // scrolls to it before the web fonts arrive, and the narrower Inter Tight then shortens the text above it; put the
  // target back under the bar once the fonts are in, unless the reader has started to move. Not on a reload or a
  // back / forward visit (CROSS.restored): the browser then restores the reader's place, which may be far from the
  // #section (the leaderboard writes the open table's #lb-tN into the address). A page script may name another element
  // to show instead (CROSS.hashTarget: the leaderboard shows a table's tabs above it).
  try {
    CROSS.hashTarget = !CROSS.restored && location.hash.length > 1 &&
      document.getElementById(decodeURIComponent(location.hash.slice(1))) || null;
  } catch (e) { CROSS.hashTarget = null; }
  if (CROSS.hashTarget) {
    var moved = false, until = Date.now() + 8000;  // the fonts arrive within seconds; later the reader decides
    ['wheel', 'touchstart', 'keydown', 'pointerdown'].forEach(function (t) {
      window.addEventListener(t, function () { moved = true; }, { passive: true, once: true });
    });
    var align = function () {
      var el = CROSS.hashTarget;
      if (!el || moved || Date.now() > until) return;
      var off = el.getBoundingClientRect().top - landing(el);
      if (Math.abs(off) < 2) return;
      var prev = root.style.scrollBehavior;
      root.style.scrollBehavior = 'auto';
      window.scrollTo(0, window.scrollY + off);
      root.style.scrollBehavior = prev;
    };
    if (document.fonts) {
      // ready can resolve before the font stylesheet has even arrived; loadingdone follows each batch of font files
      if (document.fonts.addEventListener) document.fonts.addEventListener('loadingdone', align);
      if (document.fonts.ready) document.fonts.ready.then(align);
    }
    window.addEventListener('load', align);
  }
})();
