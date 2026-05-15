/* ═══════════════════════════════════════════════════════════════════
   ML.TRADE Dashboard Charts — Plotly Helper
   ═══════════════════════════════════════════════════════════════════ */

window.DashboardCharts = (function () {
  'use strict';

  /* ── Shared palette ── */
  var COLORS = {
    crimson:  '#B83A3A',
    rose:     '#C8706A',
    blue:     '#8A97C8',
    green:    '#00C896',
    red:      '#E53E3E',
    amber:    '#F6C90E',
    muted:    '#7A7A8E',
    white:    '#FFFFFF',
    charcoal: '#131318',
    charcoalLight: '#1C1C24',
  };

  /* ── Base layout that every chart inherits ── */
  function baseLayout(overrides) {
    var base = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor:  'rgba(0,0,0,0)',
      font: {
        family: '"Inter", -apple-system, sans-serif',
        size: 11,
        color: '#A0A0B0',
      },
      margin: { t: 16, r: 24, b: 40, l: 52 },
      xaxis: {
        gridcolor:     'rgba(255,255,255,0.04)',
        zerolinecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 10, color: '#7A7A8E' },
      },
      yaxis: {
        gridcolor:     'rgba(255,255,255,0.04)',
        zerolinecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 10, color: '#7A7A8E' },
      },
      legend: {
        orientation: 'h',
        x: 0.5,
        xanchor: 'center',
        y: -0.15,
        font: { size: 10, color: '#A0A0B0' },
      },
      hoverlabel: {
        bgcolor: '#1C1C24',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: { family: '"Inter", sans-serif', size: 12, color: '#FFFFFF' },
      },
    };
    if (overrides) {
      for (var k in overrides) {
        if (overrides.hasOwnProperty(k)) base[k] = overrides[k];
      }
    }
    return base;
  }

  /* ── Render a Plotly chart ── */
  function render(el, traces, layout, config) {
    if (!el) return;
    var cfg = config || { displayModeBar: false, responsive: true };
    if (window.Plotly) {
      window.Plotly.newPlot(el, traces, layout, cfg);
    }
  }

  /* ── Training accuracy/loss curves ── */
  function drawTrainingHistory(containerId, historyJsonId) {
    var node = document.getElementById(historyJsonId || 'training-history');
    var el   = document.getElementById(containerId || 'chart-accuracy');
    if (!node || !el) return;

    var data;
    try { data = JSON.parse(node.textContent); } catch (e) { return; }
    if (!data) return;

    function epochs(series) {
      return (series || []).map(function (_, i) { return i + 1; });
    }

    // Accuracy chart
    if (data.accuracy || data.val_accuracy) {
      var accTraces = [];
      if (data.accuracy && data.accuracy.length) {
        accTraces.push({
          x: epochs(data.accuracy), y: data.accuracy,
          mode: 'lines+markers', name: 'Train',
          line: { color: COLORS.rose, width: 2 },
          marker: { size: 3 },
        });
      }
      if (data.val_accuracy && data.val_accuracy.length) {
        accTraces.push({
          x: epochs(data.val_accuracy), y: data.val_accuracy,
          mode: 'lines+markers', name: 'Validation',
          line: { color: COLORS.white, width: 2, dash: 'dot' },
          marker: { size: 3 },
        });
      }
      var accEl = document.getElementById('chart-accuracy');
      if (accEl && accTraces.length) {
        render(accEl, accTraces, baseLayout({
          yaxis: { title: 'Accuracy', gridcolor: 'rgba(255,255,255,0.04)', tickfont: { size: 10, color: '#7A7A8E' } },
          xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.04)', tickfont: { size: 10, color: '#7A7A8E' } },
          height: 260,
        }));
      }
    }

    // Loss chart
    if (data.loss || data.val_loss) {
      var lossTraces = [];
      if (data.loss && data.loss.length) {
        lossTraces.push({
          x: epochs(data.loss), y: data.loss,
          mode: 'lines+markers', name: 'Train',
          line: { color: COLORS.crimson, width: 2 },
          marker: { size: 3 },
        });
      }
      if (data.val_loss && data.val_loss.length) {
        lossTraces.push({
          x: epochs(data.val_loss), y: data.val_loss,
          mode: 'lines+markers', name: 'Validation',
          line: { color: COLORS.white, width: 2, dash: 'dot' },
          marker: { size: 3 },
        });
      }
      var lossEl = document.getElementById('chart-loss');
      if (lossEl && lossTraces.length) {
        render(lossEl, lossTraces, baseLayout({
          yaxis: { title: 'Loss', gridcolor: 'rgba(255,255,255,0.04)', tickfont: { size: 10, color: '#7A7A8E' } },
          xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.04)', tickfont: { size: 10, color: '#7A7A8E' } },
          height: 260,
        }));
      }
    }
  }

  /* ── Regime distribution donut ── */
  function drawRegimeDonut(containerId, regimeDist) {
    var el = document.getElementById(containerId);
    if (!el || !regimeDist) return;

    var labels = [], values = [], colors = [];
    var labelMap = { BULL: { color: COLORS.green }, BEAR: { color: COLORS.red }, CRISIS: { color: COLORS.amber } };

    for (var key in regimeDist) {
      if (regimeDist.hasOwnProperty(key)) {
        labels.push(key);
        values.push(regimeDist[key]);
        colors.push((labelMap[key] || {}).color || COLORS.muted);
      }
    }

    var traces = [{
      type: 'pie',
      labels: labels,
      values: values,
      marker: { colors: colors, line: { color: '#131318', width: 2 } },
      hole: 0.65,
      textinfo: 'label+percent',
      textfont: { size: 11, color: '#FFFFFF' },
      hoverinfo: 'label+percent+value',
    }];

    render(el, traces, baseLayout({
      height: 280,
      margin: { t: 16, r: 16, b: 16, l: 16 },
      showlegend: true,
      legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.05 },
    }));
  }

  /* ── OHLCV Candlestick chart ── */
  function drawOHLCV(containerId, ohlcvData) {
    var el = document.getElementById(containerId);
    if (!el || !ohlcvData || !ohlcvData.datetime) return;

    var dates = ohlcvData.datetime;
    var incColor = COLORS.green;
    var decColor = COLORS.red;

    var trace = {
      type: 'candlestick',
      x: dates,
      open:  ohlcvData.open,
      high:  ohlcvData.high,
      low:   ohlcvData.low,
      close: ohlcvData.close,
      increasing: { line: { color: incColor } },
      decreasing: { line: { color: decColor } },
      xaxis: 'x',
      yaxis: 'y',
    };

    var volTrace = {
      type: 'bar',
      x: dates,
      y: ohlcvData.volume,
      marker: { color: 'rgba(255,255,255,0.08)' },
      xaxis: 'x',
      yaxis: 'y2',
      showlegend: false,
    };

    var layout = baseLayout({
      height: 420,
      yaxis: { title: 'Price', gridcolor: 'rgba(255,255,255,0.04)', domain: [0.3, 1] },
      yaxis2: { title: 'Volume', gridcolor: 'rgba(255,255,255,0.04)', domain: [0, 0.25], showticklabels: false },
      xaxis: { rangeslider: { visible: false }, gridcolor: 'rgba(255,255,255,0.04)' },
      showlegend: false,
    });

    render(el, [trace, volTrace], layout);
  }

  /* ── Public API ── */
  return {
    render: render,
    baseLayout: baseLayout,
    COLORS: COLORS,
    drawTrainingHistory: drawTrainingHistory,
    drawRegimeDonut: drawRegimeDonut,
    drawOHLCV: drawOHLCV,
  };
})();
