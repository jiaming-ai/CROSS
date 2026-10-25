// CROSS site: leaderboard page. tools/build.py renders every row of the three tables as static HTML, with the default
// view (the best configuration of each system, ranked) visible. This script switches the task, filters the rows by
// sensor and inputs, keeps the best configuration of each system, orders and ranks the rows with build.py's rules
// (lb_key), and adds a tooltip to the overall bars.
(function () {
  'use strict';
  var board = document.getElementById('leaderboard');
  if (!board) return;
  var controls = board.querySelector('.lb-controls');
  var tabs = [].slice.call(board.querySelectorAll('[role="tab"]'));
  var panels = [].slice.call(board.querySelectorAll('.lb-panel'));
  var groups = [].slice.call(board.querySelectorAll('.lb-filters [role="radiogroup"]'));
  var status = document.getElementById('lb-status');
  var state = { task: 't3', sensor: 'all', inputs: 'all', view: 'best' };
  var UNRANKED = '<span aria-hidden="true">–</span><span class="sr-only">not ranked</span>';  // build.py LB_UNRANKED

  // ---------------------------------------------------------------- order and rank (tools/build.py lb_key)
  // rows with an overall value, best first (T1: lower is better), then T1 rows whose mean leaves failed sequences out,
  // then rows without an overall value; ties in table order
  function sortKey(tr, t1) {
    var d = tr.dataset;
    if (d.v === '') return [2, 0, 0, +d.i];
    var s = t1 ? 1 : -1;
    return [t1 && d.fail ? 1 : 0, s * +d.v, s * (d.v2 === '' ? 0 : +d.v2), +d.i];
  }
  function byKey(a, b) {
    for (var j = 0; j < 4; j++) if (a.key[j] !== b.key[j]) return a.key[j] < b.key[j] ? -1 : 1;
    return 0;
  }

  function render(panel) {
    var t1 = panel.getAttribute('data-task') === 't1';
    var tbody = panel.querySelector('tbody');
    var sep = tbody.querySelector('.lb-sep');
    var rows = [].slice.call(tbody.querySelectorAll('tr.lb-row')).map(function (tr) {
      return { tr: tr, key: sortKey(tr, t1) };
    });
    var shown = rows.filter(function (r) {
      var d = r.tr.dataset;
      return (state.sensor === 'all' || d.sensor === state.sensor) && (state.inputs === 'all' || d.inputs === state.inputs);
    }).sort(byKey);
    if (state.view === 'best') {
      var seen = {};
      shown = shown.filter(function (r) {
        var f = r.tr.dataset.family;
        return seen[f] ? false : (seen[f] = true);
      });
    }
    // equal values share a rank; the rows after the ranked ones get none
    var rank = 0, prev = null, max = 0, unranked = null;
    shown.forEach(function (r, n) {
      var cell = r.tr.querySelector('.lb-rank');
      if (r.key[0] === 0) {
        if (r.key[1] !== prev) { rank = n + 1; prev = r.key[1]; }
        cell.textContent = rank;
      } else {
        cell.innerHTML = UNRANKED;
        unranked = unranked || r.tr;
      }
      if (r.key[0] < 2) max = Math.max(max, +r.tr.dataset.v);
    });
    // T1: the longest bar shown fills the track (T2 and T3 bars are fractions of 100 %)
    if (t1) shown.forEach(function (r) {
      var bar = r.tr.querySelector('.lb-bar i');
      if (bar) bar.style.setProperty('--w', (+r.tr.dataset.v / max * 100).toFixed(1) + '%');
    });
    // the shown rows in order, with the divider above the first unranked one; then the others, hidden, in table order
    var on = shown.map(function (r) { return r.tr; });
    on.forEach(function (tr) {
      if (tr === unranked) tbody.appendChild(sep);
      tr.hidden = false;
      tbody.appendChild(tr);
    });
    sep.hidden = !unranked;
    if (!unranked) tbody.appendChild(sep);
    rows.filter(function (r) { return on.indexOf(r.tr) < 0; })
      .sort(function (a, b) { return a.key[3] - b.key[3]; })
      .forEach(function (r) { r.tr.hidden = true; tbody.appendChild(r.tr); });

    var empty = tbody.querySelector('.lb-empty');
    if (!shown.length && !empty) {
      empty = document.createElement('tr');
      empty.className = 'lb-empty';
      var td = document.createElement('td');
      td.colSpan = panel.querySelectorAll('thead th').length;
      td.textContent = 'No system in the benchmark runs with these inputs.';
      empty.appendChild(td);
      tbody.appendChild(empty);
    }
    if (empty) empty.hidden = shown.length > 0;
    panel.lbCount = shown.length;
  }

  function activePanel() {
    return panels.filter(function (p) { return p.getAttribute('data-task') === state.task; })[0];
  }
  function updateStatus() {
    var n = activePanel().lbCount;
    status.textContent = state.view === 'best'
      ? n + (n === 1 ? ' system' : ' systems') + ', the best configuration of each'
      : n + (n === 1 ? ' configuration' : ' configurations');
  }

  // ---------------------------------------------------------------- task tabs
  function tabOf(task) { return document.getElementById('lb-tab-' + task); }
  function select(task) {
    state.task = task;
    tabs.forEach(function (t) {
      var on = t.getAttribute('data-task') === task;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
    });
    panels.forEach(function (p) { p.classList.toggle('is-active', p.getAttribute('data-task') === task); });
    updateStatus();
  }
  function selectAndLink(task) {
    select(task);
    try { history.replaceState(null, '', '#lb-' + task); } catch (e) {}
  }
  tabs.forEach(function (t, k) {
    t.addEventListener('click', function () { selectAndLink(t.getAttribute('data-task')); });
    t.addEventListener('keydown', function (e) {
      var j = { ArrowRight: k + 1, ArrowLeft: k - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (j === undefined) return;
      e.preventDefault();
      var next = tabs[(j + tabs.length) % tabs.length];
      selectAndLink(next.getAttribute('data-task'));
      next.focus();
    });
  });
  // with the tabs, the selected tab names the table: its heading stays for screen readers only
  panels.forEach(function (p) {
    p.setAttribute('role', 'tabpanel');
    p.setAttribute('aria-labelledby', 'lb-tab-' + p.getAttribute('data-task'));
    p.tabIndex = 0;
    p.querySelector('.lb-title').classList.add('sr-only');
  });

  // links to a task's table (#lb-t1 ...): select its tab, move the focus to it and scroll to the controls
  document.querySelectorAll('a[href^="#lb-t"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var task = a.getAttribute('href').slice(4);
      if (!tabOf(task)) return;
      e.preventDefault();
      selectAndLink(task);
      tabOf(task).focus({ preventScroll: true });
      controls.scrollIntoView();
    });
  });

  // ---------------------------------------------------------------- filters: single-choice groups (ARIA radio groups)
  // each group is one tab stop (its chosen option); the arrow keys, Home and End move the choice; a click or Space
  // chooses. The chosen option is marked by aria-checked, which site.css's .seg draws with the accent square.
  function mark(group) {
    group.querySelectorAll('[role="radio"]').forEach(function (o) {
      var on = o.getAttribute('data-value') === state[o.getAttribute('data-filter')];
      o.setAttribute('aria-checked', String(on));
      o.tabIndex = on ? 0 : -1;
    });
  }
  function pick(group, b) {
    var f = b.getAttribute('data-filter'), v = b.getAttribute('data-value');
    if (state[f] === v) return;
    state[f] = v;
    mark(group);
    panels.forEach(render);
    updateStatus();
  }
  groups.forEach(function (group) {
    group.addEventListener('click', function (e) {
      var b = e.target.closest('[role="radio"]');
      if (b) pick(group, b);
    });
    group.addEventListener('keydown', function (e) {
      var step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      var items = [].slice.call(group.querySelectorAll('[role="radio"]'));
      var i = items.indexOf(document.activeElement), n = items.length;
      if (i < 0 || (step === undefined && e.key !== 'Home' && e.key !== 'End')) return;
      e.preventDefault();
      var b = items[e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : (i + step + n) % n];
      b.focus();
      pick(group, b);
    });
    mark(group);
  });

  // ---------------------------------------------------------------- tooltip on the overall bars (values are also in the row)
  var tip = document.createElement('div');
  tip.className = 'chart-tooltip';
  tip.hidden = true;
  tip.setAttribute('aria-hidden', 'true');  // repeats the row, which screen readers read as a table row
  document.body.appendChild(tip);
  var tipRow = null;
  function line(cls, text, small) {
    var el = document.createElement('div');
    el.className = cls;
    el.textContent = text;
    if (small) {
      var s = document.createElement('small');
      s.textContent = small;
      el.appendChild(s);
    }
    return el;
  }
  function fillTip(tr) {
    var val = tr.querySelector('.lb-val'), strict = val.querySelector('small'), cfg = tr.querySelector('.lb-cfg');
    var t1 = tr.closest('.lb-panel').getAttribute('data-task') === 't1';
    var about = [].map.call(tr.querySelectorAll('.lb-tags .badge'), function (b) { return b.textContent; });
    var ds = [].map.call(tr.querySelectorAll('.lb-mini > span'), function (s) { return s.textContent; });
    if (cfg) about.unshift(cfg.textContent);
    tip.textContent = '';
    tip.appendChild(line('lb-tip-v', val.querySelector('b').textContent, strict ? strict.textContent : t1 ? 'of the path' : ''));
    tip.appendChild(line('lb-tip-n', tr.querySelector('.lb-name').textContent, about.join(' · ')));
    tip.appendChild(line('lb-tip-ds', ds.join(' · ')));
  }
  function hideTip() { tip.hidden = true; tipRow = null; }
  board.addEventListener('pointermove', function (e) {
    var ov = e.pointerType !== 'touch' && e.target.closest && e.target.closest('.lb-ov:not(.is-na)');
    var tr = ov && ov.closest('.lb-row');
    if (!tr) { hideTip(); return; }
    if (tr !== tipRow) { fillTip(tr); tipRow = tr; tip.hidden = false; }
    var x = e.clientX + 16, y = e.clientY + 16;
    if (x + tip.offsetWidth > window.innerWidth - 8) x = e.clientX - tip.offsetWidth - 16;
    if (y + tip.offsetHeight > window.innerHeight - 8) y = e.clientY - tip.offsetHeight - 16;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  });
  board.addEventListener('pointerleave', hideTip);
  window.addEventListener('scroll', hideTip, { passive: true });

  // ---------------------------------------------------------------- start: the default view, or the task in the URL
  panels.forEach(render);
  var m = /^#lb-(t[123])$/.exec(location.hash);
  select(m ? m[1] : state.task);
  // the browser scrolls to the panel on load and focuses it (it is the link's target): show the tabs above it instead
  // (site.js puts CROSS.hashTarget under the bar, again once the fonts are in), and put the focus on the selected tab
  // rather than around the whole table. Not on a reload or a back / forward visit (CROSS.restored): the browser puts
  // the reader back where they were, and the tab in the address is only the table they had open.
  if (m && !(window.CROSS && window.CROSS.restored)) {
    if (window.CROSS) window.CROSS.hashTarget = controls;
    window.addEventListener('load', function () { tabOf(m[1]).focus({ preventScroll: true }); });
  }
})();
