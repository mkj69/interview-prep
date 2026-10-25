/* Interview Prep — shared page behaviour
 * - left rail: module list + in-page TOC
 * - speed mode (hide deep content, keep one-liners + English answers)
 * - copy buttons, reading progress, syntax highlight, KaTeX
 * - flashcards with a small Leitner-style spaced repetition store (localStorage)
 */
(function () {
  'use strict';

  var ROOT = document.documentElement.getAttribute('data-root') || '.';
  var STORE_KEY = 'iprep.srs.v1';
  var PREF_KEY = 'iprep.prefs.v1';
  var DAY = 86400000;
  // interval (days) after a "good" rating, indexed by the new box number
  var INTERVALS = [0, 2, 5, 12, 25, 50];

  /* ---------------- storage (never throws) ---------------- */
  function load(key, fallback) {
    try { var v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
    catch (e) { return fallback; }
  }
  function save(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* private mode etc. */ }
  }

  var SRS = {
    all: function () { return load(STORE_KEY, {}); },
    get: function (id) { return this.all()[id] || null; },
    rate: function (id, rating) {
      var db = this.all();
      var s = db[id] || { box: 0, due: 0, n: 0 };
      var now = Date.now();
      if (rating === 'bad') { s.box = 0; s.due = now; }
      else if (rating === 'fuzzy') { s.box = Math.max(1, Math.min(s.box, 2)); s.due = now + DAY; }
      else { s.box = Math.min(s.box + 1, INTERVALS.length - 1); s.due = now + INTERVALS[s.box] * DAY; }
      s.last = rating; s.n = (s.n || 0) + 1; s.t = now;
      db[id] = s; save(STORE_KEY, db);
      return s;
    },
    isDue: function (id) { var s = this.get(id); return !s || s.due <= Date.now(); },
    reset: function (ids) {
      var db = this.all();
      ids.forEach(function (id) { delete db[id]; });
      save(STORE_KEY, db);
    }
  };
  window.IPREP_SRS = SRS;

  /* ---------------- helpers ---------------- */
  function $(sel, el) { return (el || document).querySelector(sel); }
  function $$(sel, el) { return Array.prototype.slice.call((el || document).querySelectorAll(sel)); }
  function h(tag, attrs, html) {
    var el = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    if (html != null) el.innerHTML = html;
    return el;
  }
  function deckStats(deck) {
    var db = SRS.all(), good = 0, seen = 0, due = 0, now = Date.now();
    deck.cards.forEach(function (c) {
      var s = db[c.id];
      if (s) { seen++; if (s.last === 'good') good++; if (s.due <= now) due++; } else { due++; }
    });
    return { total: deck.cards.length, good: good, seen: seen, due: due };
  }
  window.IPREP_deckStats = deckStats;

  /* ---------------- rail ---------------- */
  function buildRail() {
    var nav = $('#module-nav');
    if (nav && window.MODULES) {
      var current = document.body.getAttribute('data-module');
      window.MODULES.forEach(function (m) {
        var a = h('a', { href: ROOT + '/' + m.file }, '<span class="num">' + m.num + '</span>' + m.title);
        if (m.id === current) a.className = 'active';
        if (m.status === 'planned') { a.className += ' soon'; a.title = '还在写'; }
        nav.appendChild(a);
      });
    }
    var toc = $('#toc');
    if (toc) {
      var heads = $$('h2.sec, .concept-head h3');
      heads.forEach(function (el, i) {
        var target = el.classList.contains('sec') ? el : el.closest('.concept');
        if (!target.id) target.id = 's' + i;
        var label = el.cloneNode(true);
        $$('.num', label).forEach(function (n) { n.remove(); });
        var a = h('a', { href: '#' + target.id }, label.textContent.trim());
        if (!el.classList.contains('sec')) a.className = 'lvl3';
        a.dataset.target = target.id;
        toc.appendChild(a);
      });
      // scroll spy
      var links = $$('a', toc);
      var targets = links.map(function (a) { return document.getElementById(a.dataset.target); });
      window.addEventListener('scroll', function () {
        var y = window.scrollY + 120, idx = 0;
        targets.forEach(function (t, i) { if (t && t.offsetTop <= y) idx = i; });
        links.forEach(function (a, i) { a.classList.toggle('here', i === idx); });
      }, { passive: true });
    }
  }

  /* ---------------- toolbar ---------------- */
  function setupToolbar() {
    var prefs = load(PREF_KEY, {});
    var speed = $('#speed-toggle');
    function applySpeed(on) {
      document.body.classList.toggle('speed', on);
      if (speed) speed.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    applySpeed(!!prefs.speed);
    if (speed) speed.addEventListener('click', function () {
      prefs.speed = !document.body.classList.contains('speed');
      save(PREF_KEY, prefs); applySpeed(prefs.speed);
    });
    var expand = $('#expand-toggle');
    if (expand) expand.addEventListener('click', function () {
      var open = expand.getAttribute('aria-pressed') !== 'true';
      $$('main details').forEach(function (d) { if (!d.closest('.fcard')) d.open = open; });
      expand.setAttribute('aria-pressed', open ? 'true' : 'false');
      expand.textContent = open ? '全部收起' : '全部展开';
    });
  }

  /* ---------------- code ---------------- */
  function setupCode() {
    $$('pre').forEach(function (pre) {
      var btn = h('button', { class: 'copy', type: 'button' }, 'copy');
      btn.addEventListener('click', function () {
        var text = (pre.querySelector('code') || pre).innerText;
        try {
          navigator.clipboard.writeText(text).then(function () {
            btn.textContent = 'copied'; setTimeout(function () { btn.textContent = 'copy'; }, 1200);
          });
        } catch (e) { /* ignore */ }
      });
      pre.appendChild(btn);
    });
    if (window.hljs) {
      $$('pre code').forEach(function (c) {
        if (!/language-/.test(c.className)) c.classList.add('language-python');
        try { window.hljs.highlightElement(c); } catch (e) { /* ignore */ }
      });
    }
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(document.body, {
          delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        });
      } catch (e) { /* ignore */ }
    }
  }

  /* ---------------- progress bar ---------------- */
  function setupProgress() {
    var bar = $('.reading-progress');
    if (!bar) return;
    window.addEventListener('scroll', function () {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
    }, { passive: true });
  }

  /* ---------------- flashcards (module page) ---------------- */
  function renderCard(card, onRate) {
    var s = SRS.get(card.id);
    var el = h('div', { class: 'fcard', 'data-id': card.id, 'data-state': s ? s.last : '' });
    el.innerHTML =
      '<div class="fq"><span class="state-dot"></span><div class="qtext">' + card.q + '</div>' +
      (card.tag ? '<span class="tag">' + card.tag + '</span>' : '') + '</div>' +
      '<button class="chip-btn reveal" type="button">先自己说一遍，再点开答案</button>' +
      '<div class="fa">' + card.a +
      '<div class="rate"><button class="r-bad" data-r="bad">不会</button>' +
      '<button class="r-fuzzy" data-r="fuzzy">模糊</button>' +
      '<button class="r-good" data-r="good">会了</button></div></div>';
    $('.reveal', el).addEventListener('click', function () { el.classList.add('open'); });
    $('.fq', el).addEventListener('click', function () { el.classList.toggle('open'); });
    $$('.rate button', el).forEach(function (b) {
      b.addEventListener('click', function () {
        var r = b.getAttribute('data-r');
        SRS.rate(card.id, r);
        el.setAttribute('data-state', r);
        $$('.rate button', el).forEach(function (x) { x.classList.toggle('on', x === b); });
        if (onRate) onRate();
      });
    });
    if (s) $$('.rate button', el).forEach(function (x) { x.classList.toggle('on', x.getAttribute('data-r') === s.last); });
    return el;
  }
  window.IPREP_renderCard = renderCard;

  function setupDecks() {
    $$('[data-deck]').forEach(function (host) {
      var deckId = host.getAttribute('data-deck');
      var deck = (window.DECKS || {})[deckId];
      if (!deck) return;
      var filter = host.getAttribute('data-section'); // optional subset
      var cards = deck.cards.filter(function (c) { return !filter || c.sec === filter; });
      var meta = h('div', { class: 'deck-meta' });
      function refresh() {
        var st = deckStats({ cards: cards });
        meta.textContent = '共 ' + st.total + ' 张 · 已掌握 ' + st.good + ' · 今天待复习 ' + st.due;
        var global = $('#deck-stat');
        if (global) { var g = deckStats(deck); global.textContent = '卡片 ' + g.good + '/' + g.total + ' 已掌握'; }
      }
      if (!host.hasAttribute('data-bare')) {
        var head = h('div', { class: 'deck-head' });
        head.appendChild(h('h3', null, host.getAttribute('data-title') || '自测卡片'));
        head.appendChild(meta);
        host.appendChild(head);
      }
      cards.forEach(function (c) { host.appendChild(renderCard(c, refresh)); });
      refresh();
    });
    if (window.renderMathInElement) {
      try { window.renderMathInElement(document.body, { delimiters: [{ left: '\\(', right: '\\)', display: false }, { left: '$$', right: '$$', display: true }], ignoredTags: ['script', 'pre', 'code'] }); } catch (e) {}
    }
    if (window.hljs) $$('.fcard pre code').forEach(function (c) { try { window.hljs.highlightElement(c); } catch (e) {} });
  }

  /* ---------------- index roadmap ---------------- */
  function setupRoadmap() {
    var host = $('#roadmap');
    if (!host || !window.MODULES) return;
    var priLabel = { 1: '★ 必须', 2: '☆ 应该', 3: '○ 加分' };
    window.PHASES.forEach(function (p) {
      var wrap = h('div', { class: 'phase' });
      wrap.appendChild(h('h3', { class: 'phase-title' }, p.title));
      wrap.appendChild(h('p', { class: 'phase-sub' }, p.sub));
      var list = h('div', { class: 'mod-list' });
      window.MODULES.filter(function (m) { return m.phase === p.id; }).forEach(function (m) {
        var deck = (window.DECKS || {})[m.id];
        var st = deck ? deckStats(deck) : null;
        var state = m.status === 'ready'
          ? '<span class="ready">可以学了</span>' + (st ? '<br>' + st.good + '/' + st.total + ' 张卡<div class="bar"><i style="width:' + (st.total ? (100 * st.good / st.total) : 0) + '%"></i></div>' : '')
          : (m.status === 'draft' ? '草稿' : '还在写');
        var title = m.status === 'planned' ? m.title : '<a href="' + m.file + '">' + m.title + '</a>';
        var row = h('div', { class: 'mod' });
        row.innerHTML = '<div class="mnum">' + m.num + '</div><div><h3>' + title +
          ' <span class="pri p' + m.pri + '">' + priLabel[m.pri] + '</span></h3><p>' + m.desc +
          '</p><div class="topics">' + m.topics + '</div></div><div class="mstate">' + state + '</div>';
        list.appendChild(row);
      });
      wrap.appendChild(list);
      host.appendChild(wrap);
    });
    var tot = $('#overall-stat');
    if (tot && window.DECKS) {
      var good = 0, all = 0, due = 0;
      Object.keys(window.DECKS).forEach(function (k) { var s = deckStats(window.DECKS[k]); good += s.good; all += s.total; due += s.due; });
      tot.textContent = '已掌握 ' + good + ' / ' + all + ' 张卡片 · 今天待复习 ' + due + ' 张';
    }
  }

  /* ---------------- review page ---------------- */
  function setupReview() {
    var stage = $('#review');
    if (!stage || !window.DECKS) return;
    var picker = $('#review-deck');
    var deckIds = Object.keys(window.DECKS);
    deckIds.forEach(function (id) {
      var m = (window.MODULES || []).filter(function (x) { return x.id === id; })[0];
      picker.appendChild(h('option', { value: id }, (m ? m.num + ' · ' + m.title : id)));
    });
    var queue = [], i = 0;
    function build() {
      var which = picker.value, onlyDue = $('#review-due').checked;
      queue = [];
      deckIds.forEach(function (id) {
        if (which !== 'all' && which !== id) return;
        window.DECKS[id].cards.forEach(function (c) {
          if (!onlyDue || SRS.isDue(c.id)) queue.push({ deck: id, card: c });
        });
      });
      // unseen + bad first, then shuffle within groups
      queue.sort(function (a, b) {
        var sa = SRS.get(a.card.id), sb = SRS.get(b.card.id);
        var ka = !sa ? 1 : (sa.last === 'bad' ? 0 : 2), kb = !sb ? 1 : (sb.last === 'bad' ? 0 : 2);
        return ka - kb || Math.random() - 0.5;
      });
      i = 0; show();
    }
    function show() {
      stage.innerHTML = '';
      var bar = $('#review-count');
      if (i >= queue.length) {
        stage.innerHTML = '<div class="review-card"><p class="q">这一轮复习完了 🎉</p><p>可以换一个模块，或者取消"只看到期的卡"再刷一遍。</p></div>';
        bar.textContent = '0 张剩余';
        return;
      }
      bar.textContent = '第 ' + (i + 1) + ' / ' + queue.length + ' 张';
      var item = queue[i];
      var m = (window.MODULES || []).filter(function (x) { return x.id === item.deck; })[0];
      var card = h('div', { class: 'review-card' });
      card.innerHTML = '<div class="src">' + (m ? m.num + ' · ' + m.title : item.deck) + (item.card.tag ? ' · ' + item.card.tag : '') + '</div>' +
        '<div class="q">' + item.card.q + '</div>' +
        '<button class="big-btn ghost" type="button" id="rv-show">先说出来，然后看答案（空格）</button>' +
        '<div class="a" hidden>' + item.card.a +
        '<div class="rate"><button class="r-bad" data-r="bad">不会（1）</button><button class="r-fuzzy" data-r="fuzzy">模糊（2）</button><button class="r-good" data-r="good">会了（3）</button></div></div>';
      stage.appendChild(card);
      if (window.renderMathInElement) { try { window.renderMathInElement(card, { delimiters: [{ left: '\\(', right: '\\)', display: false }, { left: '$$', right: '$$', display: true }], ignoredTags: ['pre', 'code'] }); } catch (e) {} }
      if (window.hljs) $$('pre code', card).forEach(function (c) { try { window.hljs.highlightElement(c); } catch (e) {} });
      $('#rv-show').addEventListener('click', reveal);
      $$('.rate button', card).forEach(function (b) {
        b.addEventListener('click', function () { rate(b.getAttribute('data-r')); });
      });
    }
    function reveal() { var a = $('.review-card .a'); if (a) { a.hidden = false; var b = $('#rv-show'); if (b) b.remove(); } }
    function rate(r) { if (i < queue.length) { SRS.rate(queue[i].card.id, r); i++; show(); } }
    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
      if (e.key === ' ') { e.preventDefault(); reveal(); }
      var a = $('.review-card .a');
      if (a && !a.hidden) {
        if (e.key === '1') rate('bad'); if (e.key === '2') rate('fuzzy'); if (e.key === '3') rate('good');
      }
    });
    picker.addEventListener('change', build);
    $('#review-due').addEventListener('change', build);
    $('#review-reset').addEventListener('click', function () {
      if (!confirm('清空所选范围内所有卡片的复习记录？')) return;
      var ids = [];
      deckIds.forEach(function (id) { if (picker.value === 'all' || picker.value === id) window.DECKS[id].cards.forEach(function (c) { ids.push(c.id); }); });
      SRS.reset(ids); build();
    });
    build();
  }

  document.addEventListener('DOMContentLoaded', function () {
    buildRail();
    setupToolbar();
    setupCode();
    setupProgress();
    setupDecks();
    setupRoadmap();
    setupReview();
  });
})();
