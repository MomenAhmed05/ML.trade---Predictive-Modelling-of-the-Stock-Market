/* ═══════════════════════════════════════════════════════════════════
   ML.TRADE Dashboard — Entrance Animations & Count-Up
   ═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  // Respect reduced-motion preference
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Number count-up for metric values ───────────────────────────── */
  function parseNumber(text) {
    // Extract the first number (supports +/- sign, commas, decimals)
    var match = text.replace(/,/g, '').match(/[-+]?\d+(\.\d+)?/);
    return match ? parseFloat(match[0]) : null;
  }

  function formatLike(template, value) {
    // Preserve the original formatting: sign, decimals, % suffix, £ prefix
    var hasPercent  = /%/.test(template);
    var hasPound    = /£/.test(template);
    var hasPlusSign = /^\s*\+/.test(template);
    var isNegative  = value < 0;
    var decimalMatch = template.match(/\.(\d+)/);
    var decimals = decimalMatch ? decimalMatch[1].length : 0;

    var absStr = Math.abs(value).toFixed(decimals);
    // Thousands separators if original had them
    if (/,/.test(template)) {
      var parts = absStr.split('.');
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      absStr = parts.join('.');
    }

    var sign = '';
    if (isNegative)           sign = '-';
    else if (hasPlusSign)     sign = '+';

    return (hasPound ? '£' : '') + sign + absStr + (hasPercent ? '%' : '');
  }

  function animateNumber(el, target, template) {
    var duration = 1200;
    var start = null;
    var startVal = 0;
    var ease = function (t) { return 1 - Math.pow(1 - t, 3); }; // easeOutCubic

    function frame(ts) {
      if (!start) start = ts;
      var p = Math.min(1, (ts - start) / duration);
      var eased = ease(p);
      var current = startVal + (target - startVal) * eased;
      el.textContent = formatLike(template, current);
      if (p < 1) {
        requestAnimationFrame(frame);
      } else {
        el.textContent = formatLike(template, target);
        el.setAttribute('data-counted', 'true');
      }
    }
    requestAnimationFrame(frame);
  }

  function initCountUp() {
    var values = document.querySelectorAll('.hero-metric-grid .metric-card__value');
    if (!values.length) return;

    if (reduced) return; // skip: text is already correct

    values.forEach(function (el, i) {
      var template = (el.textContent || '').trim();
      var target = parseNumber(template);
      if (target === null || !isFinite(target)) return;

      // Seed with zero-equivalent so the flash is visible
      el.textContent = formatLike(template, 0);

      // Stagger after the card entrance completes
      var delay = 700 + i * 80;
      setTimeout(function () { animateNumber(el, target, template); }, delay);
    });
  }

  /* ── IntersectionObserver: fade-in below-fold content ────────────── */
  function initScrollFade() {
    if (!('IntersectionObserver' in window) || reduced) return;
    var targets = document.querySelectorAll(
      '.content > .card, .content > .metric-cards, .equity-gallery, .regime-timeline'
    );
    if (!targets.length) return;

    targets.forEach(function (t) {
      t.style.opacity = '0';
      t.style.transform = 'translateY(20px)';
      t.style.transition = 'opacity 0.7s cubic-bezier(0.22, 1, 0.36, 1), transform 0.7s cubic-bezier(0.22, 1, 0.36, 1)';
    });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    targets.forEach(function (t) { io.observe(t); });
  }

  /* ── Init on DOM ready ───────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      initCountUp();
      initScrollFade();
    });
  } else {
    initCountUp();
    initScrollFade();
  }
})();
