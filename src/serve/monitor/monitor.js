'use strict';
// NInfer Monitor: polls GET /stats and GET /api/requests every 2 s; a selected request's texts come
// from GET /api/requests/<id> when it changes. Layout after Strata's API Monitor.
(() => {
  const POLL_MS = 2000;
  const SPARK_POINTS = 60;
  const $ = (id) => document.getElementById(id);

  const store = {
    get(area, key) { try { return window[area].getItem(key); } catch { return null; } },
    set(area, key, value) { try { window[area].setItem(key, value); } catch { /* storage off */ } },
  };

  let apiKey = store.get('sessionStorage', 'ninfer.monitor.key') || '';
  let selected = null;
  let pane = 'input';
  let detail = null;
  let detailKey = '';
  let listKey = '';
  let previous = null;
  let summaries = [];
  let queuedIds = new Set();
  const decodeHistory = [];
  const prefillHistory = [];

  // Theme: the stored choice, else the system's.
  const root = document.documentElement;
  const storedTheme = store.get('localStorage', 'ninfer.monitor.theme');
  root.dataset.theme = storedTheme ||
    (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  $('theme').addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    store.set('localStorage', 'ninfer.monitor.theme', root.dataset.theme);
  });

  $('key').addEventListener('click', () => { $('auth').hidden = !$('auth').hidden; });
  $('auth').addEventListener('submit', (event) => {
    event.preventDefault();
    apiKey = $('api-key').value.trim();
    store.set('sessionStorage', 'ninfer.monitor.key', apiKey);
    $('auth').hidden = true;
    poll();
  });

  const number = (value) => (value == null || Number.isNaN(value) ? '—' : Number(value).toLocaleString());
  const rate = (value) => (value == null ? '—' : value >= 100 ? value.toFixed(0) : value.toFixed(1));
  const seconds = (value) => (value == null ? '—' : value < 1 ? `${(value * 1000).toFixed(0)} ms` : `${value.toFixed(2)} s`);
  const gib = (bytes) => `${(bytes / 1073741824).toFixed(1)} GiB`;
  const clock = (ms) => new Date(ms).toLocaleTimeString();
  const percent = (part, whole) => (whole > 0 ? Math.min(100, (100 * part) / whole) : 0);
  const duration = (total) => {
    const s = Math.floor(total);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return h > 0 ? `${h} h ${m} min` : m > 0 ? `${m} min` : `${s} s`;
  };

  async function get(path) {
    const response = await fetch(path, {
      cache: 'no-store',
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    });
    if (!response.ok) {
      const error = new Error(`${path}: HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function spark(svg, values) {
    const max = Math.max(1, ...values);
    const step = 120 / (SPARK_POINTS - 1);
    const offset = SPARK_POINTS - values.length;
    const points = values.map((v, i) => `${((offset + i) * step).toFixed(1)},${(31 - (30 * v) / max).toFixed(1)}`);
    if (points.length < 2) { svg.innerHTML = ''; return; }
    const first = points[0].split(',')[0];
    svg.innerHTML =
      `<path d="M${first},32 L${points.join(' L')} L120,32 Z" fill="currentColor" opacity=".12"/>` +
      `<polyline points="${points.join(' ')}" fill="none" stroke="currentColor" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
  }

  function push(history, value) {
    history.push(value);
    if (history.length > SPARK_POINTS) history.shift();
  }

  function showStats(stats) {
    const now = performance.now();
    const counters = stats.counters || {};
    const requests = stats.requests || {};
    const capacity = stats.capacity || {};
    const occupancy = stats.occupancy || {};
    $('brand-model').textContent = stats.model || 'local inference';

    if (previous) {
      const dt = (now - previous.time) / 1000;
      const decode = Math.max(0, (counters.committed_decode_tokens - previous.decode) / dt);
      const prefill = Math.max(0, (counters.computed_prefill_tokens - previous.prefill) / dt);
      push(decodeHistory, decode);
      push(prefillHistory, prefill);
      $('decode-rate').textContent = rate(decode);
      $('prefill-rate').textContent = rate(prefill);
    }
    previous = { time: now, decode: counters.committed_decode_tokens, prefill: counters.computed_prefill_tokens };
    spark($('decode-spark'), decodeHistory);
    spark($('prefill-spark'), prefillHistory);
    $('decode-note').textContent = `${number(counters.committed_decode_tokens)} tokens since start`;
    $('prefill-note').textContent = `${number(counters.computed_prefill_tokens)} tokens since start`;

    const waiting = requests.waiting || 0;
    const prefilling = requests.prefilling || 0;
    const running = requests.running || 0;
    $('running').textContent = number(running);
    $('lanes').textContent = `/ ${number(capacity.max_concurrency)} lanes`;
    $('request-split').textContent = `${prefilling} prefilling · ${requests.decode_ready || 0} decoding · ${waiting} queued`;
    $('peak').textContent = requests.peak_admitted != null ? `Peak ${requests.peak_admitted} admitted at once` : '';

    const kv = percent(occupancy.device_main_kv_pages || 0, capacity.kv_capacity_pages || 0);
    $('kv-percent').textContent = kv.toFixed(0);
    $('kv-bar').style.width = `${kv}%`;
    $('kv-bar').parentElement.dataset.tone = kv > 90 ? 'warn' : '';
    $('kv-note').textContent =
      `${number(occupancy.device_main_kv_tokens)} of ${number(capacity.kv_capacity_tokens)} tokens · context ${number(capacity.max_context)}`;

    const states = occupancy.host_state_slots || 0;
    const slots = capacity.host_state_slots || 0;
    $('host-states').textContent = number(states);
    $('host-slots').textContent = `/ ${number(slots)} states`;
    $('host-bar').style.width = `${percent(states, slots)}%`;
    $('host-note').textContent = capacity.host_kv_bytes
      ? `KV ${gib(occupancy.host_kv_bytes || 0)} of ${gib(capacity.host_kv_bytes)} in RAM`
      : 'No host KV tier';

    const reused = counters.reused_prompt_tokens || 0;
    const computed = counters.computed_prefill_tokens || 0;
    $('reuse').textContent = (reused + computed) > 0 ? percent(reused, reused + computed).toFixed(0) : '—';
    $('reuse-note').textContent = `${number(reused)} of ${number(reused + computed)} prompt tokens from cache`;
    $('uptime').textContent = stats.uptime_seconds != null ? `Up ${duration(stats.uptime_seconds)}` : '';

    queuedIds = new Set(((stats.queue || {}).entries || []).map((entry) => entry.request_id));
    const pill = $('engine-pill');
    if (waiting > 0) {
      pill.dataset.state = 'queued';
      $('engine-state').textContent = `${waiting} queued · ${running} running`;
    } else if (prefilling > 0) {
      pill.dataset.state = 'reading';
      $('engine-state').textContent = `Prefilling · ${running} running`;
    } else if (running > 0) {
      pill.dataset.state = 'generating';
      $('engine-state').textContent = `Generating · ${running} running`;
    } else {
      pill.dataset.state = 'idle';
      $('engine-state').textContent = 'Ready';
    }
  }

  function stateOf(summary) {
    if (summary.state === 'running' && queuedIds.has(summary.id)) return 'queued';
    return summary.state;
  }

  const LABELS = { running: 'Running', queued: 'Queued', completed: 'Completed', error: 'Error', rejected: 'Rejected' };

  function badge(element, state) {
    element.className = 'st-badge';
    if (state === 'running') element.classList.add('st-badge--running');
    if (state === 'queued') element.classList.add('st-badge--queued');
    if (state === 'error' || state === 'rejected') element.classList.add('st-badge--error');
    element.textContent = LABELS[state] || state;
  }

  function matches(summary) {
    const filter = $('filter').value;
    const state = stateOf(summary);
    if (filter === 'active' && state !== 'running' && state !== 'queued') return false;
    if (filter === 'completed' && state !== 'completed') return false;
    if (filter === 'error' && state !== 'error' && state !== 'rejected') return false;
    const query = $('search').value.trim().toLowerCase();
    if (!query) return true;
    return [`#${summary.id}`, String(summary.id), summary.protocol, summary.model]
      .some((field) => field && String(field).toLowerCase().includes(query));
  }

  function meta(summary) {
    const parts = [clock(summary.started_unix_ms)];
    if (summary.state === 'running') {
      parts.push(`for ${duration((Date.now() - summary.started_unix_ms) / 1000)}`);
    } else if (summary.total_seconds != null) {
      parts.push(seconds(summary.total_seconds));
    }
    if (summary.completion_tokens != null) parts.push(`${number(summary.completion_tokens)} tok`);
    if (summary.decode_tokens_per_second) parts.push(`${rate(summary.decode_tokens_per_second)} tok/s`);
    if (summary.http_status) parts.push(`HTTP ${summary.http_status}`);
    return parts.join(' · ');
  }

  function showList() {
    const shown = summaries.filter(matches);
    const key = JSON.stringify([shown.map((s) => [s.id, stateOf(s), s.finished_unix_ms]), selected,
      shown.some((s) => s.state === 'running') ? Math.floor(Date.now() / 1000) : 0]);
    if (key === listKey) return;
    listKey = key;
    const list = $('requests');
    const scroll = list.scrollTop;
    list.replaceChildren();
    if (shown.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-list';
      empty.textContent = summaries.length ? 'No request matches.' : 'No requests yet.';
      list.append(empty);
    }
    for (const summary of shown) {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'request';
      item.setAttribute('aria-selected', String(summary.id === selected));
      const top = document.createElement('div');
      top.className = 'request__top';
      const title = document.createElement('span');
      title.className = 'request__title';
      title.textContent = `#${summary.id} · ${summary.protocol || 'request'}`;
      const state = document.createElement('span');
      badge(state, stateOf(summary));
      top.append(title, state);
      const line = document.createElement('div');
      line.className = 'request__meta';
      line.textContent = meta(summary);
      item.append(top, line);
      item.addEventListener('click', () => { selected = summary.id; detailKey = ''; listKey = ''; showList(); refreshDetail(); });
      list.append(item);
    }
    list.scrollTop = scroll;
  }

  async function refreshDetail() {
    const summary = summaries.find((s) => s.id === selected);
    $('no-selection').hidden = summary != null;
    $('selection').hidden = summary == null;
    if (!summary) return;
    const key = `${summary.id}:${summary.state}:${summary.finished_unix_ms}`;
    if (key !== detailKey) {
      try {
        detail = await get(`/api/requests/${summary.id}`);
        detailKey = key;
      } catch (error) {
        if (error.status === 404) { detail = { ...summary, input: '', output: '', reasoning: '', record: null }; detailKey = key; }
      }
    }
    if (!detail || detail.id !== summary.id) return;
    showDetail({ ...detail, ...summary });
  }

  function showDetail(d) {
    $('detail-title').textContent = `Request #${d.id} · ${d.protocol || ''}`;
    badge($('detail-status'), stateOf(d));
    $('detail-time').textContent =
      `${new Date(d.started_unix_ms).toLocaleString()} · ${d.model || ''} · ${number(d.message_count)} ${d.message_count === 1 ? 'message' : 'messages'}`;
    const spec = d.speculative || {};
    const drafted = (spec.drafted_tokens || 0) + (spec.copy_drafted_tokens || 0);
    const accepted = (spec.accepted_tokens || 0) + (spec.copy_accepted_tokens || 0);
    $('f-total').textContent = seconds(d.total_seconds);
    $('f-queue').textContent = seconds(d.queue_wait_seconds);
    $('f-ttft').textContent = seconds(d.ttft_seconds);
    $('f-decode').textContent = d.decode_tokens_per_second ? `${rate(d.decode_tokens_per_second)} tok/s` : '—';
    $('f-prompt').textContent = number(d.prompt_tokens);
    $('f-cached').textContent = d.cached_tokens != null ? number(d.cached_tokens) : '—';
    $('f-prefill').textContent = d.prefill_tokens_per_second ? `${rate(d.prefill_tokens_per_second)} tok/s` : '—';
    $('f-output').textContent = d.completion_tokens != null
      ? `${number(d.completion_tokens)}${d.reasoning_tokens ? ` (${number(d.reasoning_tokens)} thinking)` : ''}` : '—';
    $('f-spec').textContent = drafted > 0 ? `${number(accepted)} / ${number(drafted)} (${percent(accepted, drafted).toFixed(0)}%)` : '—';
    $('f-path').textContent = d.prefix_reuse_path || '—';
    $('f-finish').textContent = d.finish_reason || '—';
    $('f-mode').textContent = d.stream ? 'Stream' : 'JSON';
    const error = d.error && (d.error.message || d.error.type);
    $('detail-error').hidden = !error;
    $('detail-error').textContent = error ? `${d.error.type || 'error'}: ${d.error.message || ''}` : '';
    for (const tab of document.querySelectorAll('[data-pane]')) {
      tab.setAttribute('aria-selected', String(tab.dataset.pane === pane));
    }
    let text = '';
    let note = '';
    if (pane === 'input') {
      text = d.input || '';
      if (text.startsWith('[earlier input omitted]')) note = 'Showing the latest 64 KiB of the input.';
    } else if (pane === 'output') {
      text = d.output || '';
      for (const call of d.tool_calls || []) text += `${text ? '\n\n' : ''}-> ${call.name}(${call.arguments})`;
      if (d.output_omitted_bytes) note = `Showing the first 64 KiB; ${number(d.output_omitted_bytes)} bytes more.`;
      if (!text && d.state === 'running') note = 'The output appears when the request finishes.';
    } else if (pane === 'reasoning') {
      text = d.reasoning || '';
      if (d.reasoning_omitted_bytes) note = `Showing the first 64 KiB; ${number(d.reasoning_omitted_bytes)} bytes more.`;
      if (!text) note = 'No reasoning.';
    } else {
      text = d.record ? JSON.stringify(d.record, null, 2) : '';
      if (!text) note = 'The request_done record appears when the request finishes.';
    }
    $('content-note').textContent = note;
    const content = $('content');
    if (content.textContent !== text) content.textContent = text;
  }

  for (const tab of document.querySelectorAll('[data-pane]')) {
    tab.addEventListener('click', () => { pane = tab.dataset.pane; refreshDetail(); });
  }
  $('search').addEventListener('input', () => { listKey = ''; showList(); });
  $('filter').addEventListener('change', () => { listKey = ''; showList(); });
  $('copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('content').textContent);
      $('copy').textContent = 'Copied';
    } catch {
      $('copy').textContent = 'Copy failed';
    }
    setTimeout(() => { $('copy').textContent = 'Copy'; }, 1200);
  });

  function showError(message) {
    $('error').hidden = !message;
    $('error').textContent = message || '';
  }

  let timer = null;
  async function poll() {
    clearTimeout(timer);
    try {
      const [stats, history] = await Promise.all([get('/stats'), get('/api/requests')]);
      showError('');
      showStats(stats);
      summaries = history.requests || [];
      $('retention').textContent =
        `The last ${history.capacity} requests stay in memory until the server stops. Inputs and outputs stay on this machine.`;
      showList();
      await refreshDetail();
      $('updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
    } catch (error) {
      const pill = $('engine-pill');
      pill.dataset.state = 'down';
      if (error.status === 401) {
        $('auth').hidden = false;
        $('engine-state').textContent = 'API key needed';
        showError('This server requires its API key.');
      } else if (error.status === 503) {
        $('engine-state').textContent = 'Loading model';
        showError('');
      } else {
        $('engine-state').textContent = 'Server unreachable';
        showError(`Cannot reach the server (${error.message}).`);
      }
      previous = null;
    }
    timer = setTimeout(poll, POLL_MS);
  }
  poll();
})();
