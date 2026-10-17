/* Gráfico de espectros (linhas + faixa de ±desvio) para qualquer número de bandas.
 *
 * spectrumChart(container, {
 *   x: [nm, ...] | null,          // comprimentos de onda (crescentes); null = eixo por índice de banda
 *   labels: ['365 nm', ...],      // rótulo de cada banda (tooltip / eixo por índice)
 *   series: [{ label, color, mean: [...], std: [...] | null, n }],
 *   yLabel: 'reflectância (%)',
 *   ariaLabel: '...'
 * })
 * Cores por papel (CSS var), texto sempre na tinta neutra; legenda só com >= 2 séries.
 */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const fmt = (v, d = 1) => v == null ? '—' : v.toFixed(d).replace('.', ',');

  function el(tag, attrs, parent, style) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (style) Object.assign(n.style, style);
    if (parent) parent.appendChild(n);
    return n;
  }

  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }

  function ticks(lo, hi, target) {
    const step = niceStep(hi - lo || 1, target);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }

  window.spectrumChart = function (container, opts) {
    container.querySelectorAll('.sc-root').forEach(n => n.remove());
    const root = document.createElement('div');
    root.className = 'sc-root';
    root.style.cssText = 'position:relative;width:100%;height:100%;display:flex;flex-direction:column;';
    container.appendChild(root);

    const series = opts.series || [];
    const nb = (opts.labels || []).length || (series[0] ? series[0].mean.length : 0);
    if (series.length >= 2) {
      const lg = document.createElement('div');
      lg.className = 'sc-legend';
      lg.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.74rem;color:var(--muted);padding:4px 6px 2px;';
      series.forEach(s => {
        const it = document.createElement('span');
        it.style.cssText = 'display:inline-flex;align-items:center;gap:6px;';
        const sw = document.createElement('span');
        sw.style.cssText = `width:14px;height:3px;border-radius:2px;background:${s.color};display:inline-block;`;
        it.appendChild(sw);
        it.appendChild(document.createTextNode(s.label + (s.n != null ? ` (n=${s.n})` : '')));
        lg.appendChild(it);
      });
      root.appendChild(lg);
    }
    const plot = document.createElement('div');
    plot.style.cssText = 'position:relative;flex:1;min-height:120px;';
    root.appendChild(plot);
    const W = plot.clientWidth || 300, H = plot.clientHeight || 160;
    const m = { l: 42, r: 12, t: 10, b: 24 };

    const useWl = Array.isArray(opts.x) && opts.x.length === nb;
    const xs = useWl ? opts.x : Array.from({ length: nb }, (_, i) => i);
    const x0 = xs[0], x1 = xs[nb - 1] === x0 ? x0 + 1 : xs[nb - 1];
    const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    let ymax = 0;
    series.forEach(s => s.mean.forEach((v, i) => { if (v != null) ymax = Math.max(ymax, v + ((s.std && s.std[i]) || 0)); }));
    const yt = ticks(0, ymax > 0 ? ymax * 1.05 : 1, 4);
    const ytop = Math.max(yt[yt.length - 1], ymax);
    const Y = v => m.t + (1 - v / ytop) * (H - m.t - m.b);

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
                            'aria-label': opts.ariaLabel || 'Curva espectral' }, plot, { display: 'block' });
    const axisText = { fill: 'var(--viz-muted)', fontSize: '10px', fontVariantNumeric: 'tabular-nums' };
    yt.forEach(v => {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, svg, { stroke: v === 0 ? 'var(--viz-axis)' : 'var(--viz-grid)', strokeWidth: 1 });
      el('text', { x: m.l - 6, y: Y(v) + 3, 'text-anchor': 'end' }, svg, axisText).textContent = fmt(v, v < 10 && v % 1 ? 1 : 0);
    });
    if (useWl) {
      ticks(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 70))).forEach(v => {
        if (v < x0 || v > x1) return;
        el('text', { x: X(v), y: H - 6, 'text-anchor': 'middle' }, svg, axisText).textContent = v + ' nm';
      });
    } else {
      const k = Math.max(1, Math.ceil(nb / Math.max(2, Math.floor((W - m.l - m.r) / 60))));
      for (let i = 0; i < nb; i += k) {
        const t = el('text', { x: X(i), y: H - 6, 'text-anchor': 'middle' }, svg, axisText);
        t.textContent = (opts.labels[i] || String(i + 1)).slice(0, 10);
      }
    }
    if (opts.yLabel) {
      el('text', { x: m.l + 4, y: m.t + 9 }, svg, { ...axisText, fill: 'var(--viz-muted)' }).textContent = opts.yLabel;
    }

    // faixa ±desvio, depois a linha (interrompida onde não há valor)
    series.forEach(s => {
      if (s.std) {
        let seg = [];
        const flush = () => {
          if (seg.length > 1) {
            const up = seg.map(i => `${X(xs[i]).toFixed(1)},${Y(Math.min(ytop, s.mean[i] + s.std[i])).toFixed(1)}`);
            const dn = seg.slice().reverse().map(i => `${X(xs[i]).toFixed(1)},${Y(Math.max(0, s.mean[i] - s.std[i])).toFixed(1)}`);
            el('path', { d: 'M' + up.join('L') + 'L' + dn.join('L') + 'Z' }, svg, { fill: s.color, opacity: 0.14, stroke: 'none' });
          }
          seg = [];
        };
        s.mean.forEach((v, i) => { if (v == null || s.std[i] == null) flush(); else seg.push(i); });
        flush();
      }
      let d = '', pen = false;
      s.mean.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
        pen = true;
      });
      if (d) el('path', { d }, svg, { fill: 'none', stroke: s.color, strokeWidth: 2, strokeLinejoin: 'round', strokeLinecap: 'round' });
      if (nb <= 30) s.mean.forEach((v, i) => {
        if (v != null) el('circle', { cx: X(xs[i]), cy: Y(v), r: 4 }, svg, { fill: s.color, stroke: 'var(--viz-surface)', strokeWidth: 2 });
      });
    });

    // cursor: linha vertical + pontos + tooltip com o valor de cada série na banda mais próxima
    const cross = el('line', { y1: m.t, y2: H - m.b }, svg, { stroke: 'var(--viz-axis)', strokeWidth: 1, display: 'none' });
    const dots = series.map(s => el('circle', { r: 4.5 }, svg, { fill: s.color, stroke: 'var(--viz-surface)', strokeWidth: 2, display: 'none' }));
    const tip = document.createElement('div');
    tip.style.cssText = 'position:absolute;pointer-events:none;background:var(--ink);color:var(--panel);font-size:.72rem;' +
      'padding:5px 8px;border-radius:6px;white-space:nowrap;font-variant-numeric:tabular-nums;display:none;z-index:5;line-height:1.45;';
    plot.appendChild(tip);
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, svg, { fill: 'transparent' });
    hit.addEventListener('mousemove', e => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) * W / r.width;
      let best = 0;
      xs.forEach((v, i) => { if (Math.abs(X(v) - px) < Math.abs(X(xs[best]) - px)) best = i; });
      const cx = X(xs[best]);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.display = '';
      const lines = [`<b>${escapeHtml(opts.labels[best] || ('banda ' + (best + 1)))}</b>`];
      series.forEach((s, k) => {
        const v = s.mean[best];
        if (v == null) { dots[k].style.display = 'none'; }
        else { dots[k].setAttribute('cx', cx); dots[k].setAttribute('cy', Y(v)); dots[k].style.display = ''; }
        const sd = s.std && s.std[best] != null ? ` ± ${fmt(s.std[best])}` : '';
        lines.push((series.length > 1 ? `<span style="color:${s.color}">●</span> ${escapeHtml(s.label)}: ` : '') +
                   (v == null ? 'sem valor' : fmt(v) + sd));
      });
      tip.innerHTML = lines.join('<br>');
      tip.style.display = 'block';
      const tw = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(W - tw, cx * r.width / W - tw / 2)) + 'px';
      tip.style.top = '0px';
    });
    hit.addEventListener('mouseleave', () => { cross.style.display = 'none'; tip.style.display = 'none'; dots.forEach(d => d.style.display = 'none'); });
  };

  function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  window.SERIES_COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'];
})();
