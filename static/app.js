/* ═══════════════════════════════════════════════════════════
   AI Pipeline Recovery openENV Environment Playground — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

// ── State ───────────────────────────────────────────────────
let currentSession = null;
let currentObs = null;
let episodeDone = false;
let simEventSource = null;
let stepHistory = [];

// ── Tab Switching ───────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
    });
});

// ── Mode Switching (Manual / LLM) ───────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.mode-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.mode + '-mode').classList.add('active');

        const slider = document.getElementById('mode-slider');
        if (btn.dataset.mode === 'llm') {
            slider.classList.add('right');
        } else {
            slider.classList.remove('right');
        }
    });
});

// ── README Loading ──────────────────────────────────────────
let readmeExpanded = false;

async function loadReadme() {
    try {
        const res = await fetch('/api/readme');
        const data = await res.json();
        if (data.content && typeof marked !== 'undefined') {
            document.getElementById('readme-body').innerHTML = marked.parse(data.content);
        } else {
            document.getElementById('readme-body').innerHTML = '<p>' + (data.content || 'No README found.') + '</p>';
        }
    } catch (e) {
        document.getElementById('readme-body').innerHTML = '<p class="loading-text">Failed to load README.</p>';
    }
}

function toggleReadme() {
    const container = document.getElementById('readme-content');
    const btn = document.getElementById('toggle-readme');
    readmeExpanded = !readmeExpanded;
    if (readmeExpanded) {
        container.classList.remove('collapsed');
        container.classList.add('expanded');
        btn.innerHTML = '<span class="toggle-icon">▼</span> Collapse';
    } else {
        container.classList.remove('expanded');
        container.classList.add('collapsed');
        btn.innerHTML = '<span class="toggle-icon">▶</span> Expand';
    }
}

// Load README on page load
loadReadme();

// ── Manual Mode: Reset ──────────────────────────────────────
async function resetEnv() {
    const taskId = parseInt(document.getElementById('manual-task').value);
    const seed = parseInt(document.getElementById('manual-seed').value) || 42;

    const btn = document.getElementById('btn-reset');
    btn.innerHTML = '<span class="loading-spinner"></span> Resetting…';
    btn.disabled = true;

    try {
        const res = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId, seed: seed })
        });
        const data = await res.json();

        currentSession = data.session_id;
        currentObs = data.observation;
        episodeDone = false;
        stepHistory = [];

        // Show panels
        document.getElementById('manual-initial').style.display = 'none';
        document.getElementById('obs-panel').style.display = 'block';
        document.getElementById('actions-card').style.display = 'block';
        document.getElementById('grade-panel').style.display = 'none';
        document.getElementById('task-info-card').style.display = 'block';

        // Clear history
        document.getElementById('history-body').innerHTML = '';

        // Set task info
        renderTaskInfo(data.task_config);

        // Update observation display
        updateDisplay(data.observation);

        // Clear reward panel
        document.getElementById('reward-details').innerHTML = '<div class="reward-empty">Take an action to see reward details</div>';

        enableActions(true);
    } catch (e) {
        console.error('Reset failed:', e);
        alert('Failed to reset environment: ' + e.message);
    } finally {
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/></svg> Reset Environment`;
        btn.disabled = false;
    }
}

// ── Manual Mode: Step ───────────────────────────────────────
async function stepAction(actionType) {
    if (!currentSession || episodeDone) return;

    enableActions(false);

    try {
        const res = await fetch('/api/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSession, action_type: actionType })
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            enableActions(true);
            return;
        }

        currentObs = data.observation;

        // Add to history
        const logEntry = data.episode_log && data.episode_log.length > 0
            ? data.episode_log[data.episode_log.length - 1]
            : {};
        stepHistory.push({ action: actionType, observation: data.observation, log: logEntry });

        updateDisplay(data.observation);
        addHistoryRow(actionType, data.observation, logEntry);
        updateRewardDisplay(logEntry);

        if (data.observation.done) {
            episodeDone = true;
            enableActions(false);
            if (data.grade) {
                showGrade(data.grade);
            }
            // Show done banner
            const completed = data.observation.progress_hint >= 1.0;
            showDoneBanner(completed);
        } else {
            enableActions(true);
        }
    } catch (e) {
        console.error('Step failed:', e);
        enableActions(true);
    }
}

// ── Display Helpers ─────────────────────────────────────────
function updateDisplay(obs) {
    // Status bar
    const resultEl = document.getElementById('status-result');
    resultEl.textContent = obs.tool_result;
    resultEl.className = 'status-chip result-' + obs.tool_result.toLowerCase();

    const stepEl = document.getElementById('status-step');
    stepEl.textContent = `Step ${obs.step_count}`;

    const toolEl = document.getElementById('status-tool');
    toolEl.textContent = obs.active_tool;
    toolEl.className = 'status-chip' + (obs.active_tool === 'backup' ? ' tool-backup' : '');

    const errorEl = document.getElementById('status-error');
    errorEl.textContent = obs.error_type;
    errorEl.className = 'status-chip error-' + obs.error_type.toLowerCase();

    // Budget bar
    const budgetPct = Math.round(obs.budget_remaining * 100);
    document.getElementById('bar-budget').style.width = budgetPct + '%';
    document.getElementById('val-budget').textContent = obs.budget_remaining.toFixed(2) + ` (${budgetPct}%)`;

    // Progress bar
    const progressPct = Math.round(obs.progress_hint * 100);
    document.getElementById('bar-progress').style.width = progressPct + '%';
    document.getElementById('val-progress').textContent = progressPct + '%';

    // Observation details
    const detailsEl = document.getElementById('obs-details');
    detailsEl.innerHTML = '';
    const fields = [
        ['Result', obs.tool_result],
        ['Error Type', obs.error_type],
        ['Same Errors', obs.same_error_count],
        ['Active Tool', obs.active_tool],
        ['Cooldown', obs.cooldown_remaining],
        ['Step', obs.step_count],
        ['Budget', obs.budget_remaining.toFixed(2)],
        ['Progress', (obs.progress_hint * 100).toFixed(0) + '%'],
        ['Last Error', obs.last_action_error ? 'Yes' : 'No'],
        ['Reward', obs.reward !== undefined ? obs.reward.toFixed(3) : '—'],
    ];
    fields.forEach(([label, value]) => {
        const div = document.createElement('div');
        div.className = 'detail-item';
        div.innerHTML = `<span class="detail-label">${label}</span><span class="detail-value">${value}</span>`;
        detailsEl.appendChild(div);
    });

    // Decision context
    if (obs.decision_context && obs.decision_context !== 'no active constraints detected') {
        document.getElementById('decision-context-panel').style.display = 'block';
        document.getElementById('decision-context-text').textContent = obs.decision_context;
    } else {
        document.getElementById('decision-context-panel').style.display = 'none';
    }
}

function addHistoryRow(action, obs, log) {
    const tbody = document.getElementById('history-body');
    const tr = document.createElement('tr');

    const resultClass = 'result-' + obs.tool_result.toLowerCase();
    const actionClass = action.toLowerCase();
    const rewardVal = obs.reward !== undefined ? obs.reward : 0;
    const rewardClass = rewardVal >= 0 ? 'reward-positive' : 'reward-negative';

    tr.innerHTML = `
        <td>${obs.step_count}</td>
        <td class="action-cell ${actionClass}">${action}</td>
        <td class="${resultClass}">${obs.tool_result}</td>
        <td>${obs.error_type}</td>
        <td class="${rewardClass}">${rewardVal >= 0 ? '+' : ''}${rewardVal.toFixed(3)}</td>
        <td>${obs.budget_remaining.toFixed(2)}</td>
        <td>${(obs.progress_hint * 100).toFixed(0)}%</td>
    `;
    tbody.appendChild(tr);
    tr.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateRewardDisplay(logEntry) {
    const el = document.getElementById('reward-details');
    if (!logEntry || logEntry.reward === undefined) {
        el.innerHTML = '<div class="reward-empty">No reward data available</div>';
        return;
    }

    const reward = logEntry.reward;
    const items = [
        ['Tool Result', logEntry.tool_result || '—'],
        ['Error Type', logEntry.observed_error_type || '—'],
        ['Progress', logEntry.progress !== undefined ? (logEntry.progress * 100).toFixed(0) + '%' : '—'],
        ['Bad Retry', logEntry.bad_retry ? '⚠ Yes' : 'No'],
        ['Resolved Ambiguity', logEntry.resolved_ambiguity ? '✓ Yes' : 'No'],
        ['Completed', logEntry.completed ? '✓ Yes' : 'No'],
    ];

    let html = '';
    items.forEach(([label, value]) => {
        html += `<div class="reward-row">
            <span class="reward-label">${label}</span>
            <span class="reward-value">${value}</span>
        </div>`;
    });

    const rewardClass = reward >= 0 ? 'positive' : 'negative';
    html += `<div class="reward-row total">
        <span class="reward-label">Step Reward</span>
        <span class="reward-value ${rewardClass}">${reward >= 0 ? '+' : ''}${reward.toFixed(3)}</span>
    </div>`;

    el.innerHTML = html;
}

function renderTaskInfo(config) {
    const el = document.getElementById('task-info-body');
    const fields = [
        ['Name', config.name],
        ['Max Steps', config.max_steps],
        ['Budget', config.initial_budget],
        ['Noise', (config.noise_level * 100).toFixed(0) + '%'],
        ['Ambiguity', (config.ambiguity_rate * 100).toFixed(0) + '%'],
        ['Rate Limit', config.allow_rate_limit ? 'Yes' : 'No'],
        ['Drift', config.drift_enabled ? `After step ${config.drift_after_step}` : 'No'],
        ['Cascade Pen.', config.cascade_penalty.toFixed(2)],
    ];

    el.innerHTML = fields.map(([l, v]) =>
        `<div class="info-row"><span class="info-label">${l}</span><span class="info-value">${v}</span></div>`
    ).join('');
}

function showGrade(grade) {
    const panel = document.getElementById('grade-panel');
    panel.style.display = 'block';

    const row = document.getElementById('grade-row');
    const items = [
        ['Total', grade.total, true],
        ['Completion', grade.completion, false],
        ['Efficiency', grade.efficiency, false],
        ['Cost', grade.cost, false],
        ['Recovery', grade.recovery_quality, false],
    ];

    row.innerHTML = items.map(([label, value, isTotal]) => {
        const colorClass = value >= 0.7 ? 'grade-color-high' : value >= 0.4 ? 'grade-color-mid' : 'grade-color-low';
        return `<div class="grade-item ${isTotal ? 'total-grade' : ''}">
            <div class="grade-label">${label}</div>
            <div class="grade-value ${isTotal ? '' : colorClass}">${value.toFixed(3)}</div>
        </div>`;
    }).join('');

    panel.scrollIntoView({ behavior: 'smooth' });
}

function showDoneBanner(completed) {
    const bar = document.getElementById('obs-status-bar');
    const existing = bar.querySelector('.done-banner');
    if (existing) existing.remove();

    const div = document.createElement('div');
    div.className = completed ? 'done-banner' : 'done-banner failed';
    div.textContent = completed ? '✅ Episode Complete — Task Successful!' : '❌ Episode Ended — Task Not Completed';
    bar.parentNode.insertBefore(div, bar);
}

function enableActions(enabled) {
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.disabled = !enabled;
    });
}

// ── LLM Simulation Mode ────────────────────────────────────
function startSimulation() {
    const taskIds = document.getElementById('llm-tasks').value;
    const seed = parseInt(document.getElementById('llm-seed').value) || 42;
    const useLlm = document.getElementById('llm-agent-type').value === 'llm';

    // Reset UI
    resetSimUI();

    // Show/hide buttons
    document.getElementById('btn-run-sim').style.display = 'none';
    document.getElementById('btn-stop-sim').style.display = 'flex';
    document.getElementById('sim-results-panel').style.display = 'none';

    // Start SSE
    const url = `/api/llm/stream?task_ids=${taskIds}&seed=${seed}&use_llm=${useLlm}`;
    simEventSource = new EventSource(url);

    simEventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleSimEvent(data);
        } catch (e) {
            console.error('Parse error:', e);
        }
    };

    simEventSource.onerror = () => {
        simEventSource.close();
        simEventSource = null;
        document.getElementById('btn-run-sim').style.display = 'flex';
        document.getElementById('btn-stop-sim').style.display = 'none';
    };
}

function stopSimulation() {
    if (simEventSource) {
        simEventSource.close();
        simEventSource = null;
    }
    document.getElementById('btn-run-sim').style.display = 'flex';
    document.getElementById('btn-stop-sim').style.display = 'none';
    addLogEntry('info', '—', 'Simulation stopped by user');
}

function resetSimUI() {
    // Reset task cards
    [1, 2, 3].forEach(id => {
        const card = document.getElementById('sim-task-' + id);
        if (card) {
            card.className = 'sim-task-card';
            const status = document.getElementById('sim-status-' + id);
            if (status) { status.textContent = 'Pending'; status.className = 'sim-task-status'; }
            const bar = document.getElementById('sim-bar-' + id);
            if (bar) bar.style.width = '0%';
            const meta = document.getElementById('sim-meta-' + id);
            if (meta) meta.textContent = '—';
        }
    });

    // Clear log
    document.getElementById('sim-log').innerHTML = '';
}

function handleSimEvent(data) {
    const type = data.type;

    if (type === 'task_start') {
        const card = document.getElementById('sim-task-' + data.task_id);
        if (card) card.classList.add('running');
        const status = document.getElementById('sim-status-' + data.task_id);
        if (status) { status.textContent = 'Running'; status.className = 'sim-task-status running'; }

        addLogEntry('info', data.task_name, `▶ Task ${data.task_id} started — ${data.goal}`);
        addLogEntry('step', data.task_name, `  Budget: ${data.initial_budget} | Max Steps: ${data.max_steps}`);
    }

    else if (type === 'step') {
        const obs = data.observation;
        const action = data.action;
        const reward = obs.reward;
        const rewardStr = (reward >= 0 ? '+' : '') + reward.toFixed(3);

        // Update progress bar
        const progressPct = Math.round(obs.progress_hint * 100);
        const bar = document.getElementById('sim-bar-' + data.task_id);
        if (bar) bar.style.width = progressPct + '%';

        // Update meta
        const meta = document.getElementById('sim-meta-' + data.task_id);
        if (meta) meta.textContent = `Step ${data.step}/${data.max_steps} | Budget: ${obs.budget_remaining.toFixed(2)} | Progress: ${progressPct}%`;

        // Log entry
        const resultClass = obs.tool_result === 'SUCCESS' ? 'success' : obs.tool_result === 'ERROR' ? 'error' : 'warn';
        const actionLower = action.toLowerCase();

        addLogEntryRich(resultClass, data.task_name,
            `Step ${data.step}`,
            action, actionLower,
            obs.tool_result, obs.tool_result.toLowerCase(),
            `${obs.error_type} | reward: ${rewardStr} | budget: ${obs.budget_remaining.toFixed(2)} | progress: ${progressPct}%`
        );

        // Show observation details in log for key events
        if (obs.tool_result === 'AMBIGUOUS') {
            addLogEntry('warn', data.task_name, `  ⚡ Ambiguous outcome — partial progress recorded`);
        }
        if (obs.done) {
            const completed = obs.progress_hint >= 1.0;
            addLogEntry(completed ? 'success' : 'error', data.task_name,
                completed ? '  ✅ Task completed successfully' : `  ❌ Episode ended (progress: ${progressPct}%)`
            );
        }
    }

    else if (type === 'task_complete') {
        const card = document.getElementById('sim-task-' + data.task_id);
        if (card) { card.classList.remove('running'); card.classList.add('complete'); }
        const status = document.getElementById('sim-status-' + data.task_id);
        if (status) { status.textContent = '✓ Done'; status.className = 'sim-task-status complete'; }
        const bar = document.getElementById('sim-bar-' + data.task_id);
        if (bar) bar.style.width = '100%';
        const meta = document.getElementById('sim-meta-' + data.task_id);
        if (meta) meta.textContent = `Score: ${data.grade.total.toFixed(3)} | Steps: ${data.steps}`;

        addLogEntry('grade', data.task_name,
            `🏆 Grade: ${data.grade.total.toFixed(3)} (completion: ${data.grade.completion.toFixed(2)}, efficiency: ${data.grade.efficiency.toFixed(2)}, cost: ${data.grade.cost.toFixed(2)}, recovery: ${data.grade.recovery_quality.toFixed(2)})`
        );
        addLogEntry('step', '—', '────────────────────────────────');
    }

    else if (type === 'all_complete') {
        // Show results panel
        showSimResults(data.results);

        // Stop SSE
        if (simEventSource) {
            simEventSource.close();
            simEventSource = null;
        }
        document.getElementById('btn-run-sim').style.display = 'flex';
        document.getElementById('btn-stop-sim').style.display = 'none';

        addLogEntry('info', '—', '✨ All tasks completed!');
    }

    else if (type === 'error') {
        addLogEntry('error', '—', `⚠ Error: ${data.message}`);
    }
}

function addLogEntry(type, task, msg) {
    const log = document.getElementById('sim-log');
    // Remove empty placeholder
    const empty = log.querySelector('.log-empty');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = `log-entry log-${type}`;

    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    div.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-task">[${task}]</span>
        <span class="log-msg">${msg}</span>
    `;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

function addLogEntryRich(type, task, stepText, action, actionClass, result, resultClass, details) {
    const log = document.getElementById('sim-log');
    const empty = log.querySelector('.log-empty');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = `log-entry log-${type}`;

    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    div.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-task">[${task}]</span>
        <span class="log-msg">
            ${stepText}:
            <span class="log-action log-action-${actionClass}">${action}</span>
            →
            <span class="log-result-${resultClass}">${result}</span>
            | ${details}
        </span>
    `;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

function showSimResults(results) {
    const panel = document.getElementById('sim-results-panel');
    panel.style.display = 'block';

    const grid = document.getElementById('sim-results-grid');
    let html = '';

    let totalScore = 0;
    let totalSteps = 0;

    results.forEach(r => {
        const score = r.grade.total;
        totalScore += score;
        totalSteps += r.steps;
        const pct = Math.round(score * 100);
        const colorClass = score >= 0.7 ? 'grade-color-high' : score >= 0.4 ? 'grade-color-mid' : 'grade-color-low';
        const barColor = score >= 0.7 ? 'var(--success)' : score >= 0.4 ? 'var(--warning)' : 'var(--error)';

        html += `
            <div class="sim-result-row">
                <span class="sim-result-name">${r.task_name.charAt(0).toUpperCase() + r.task_name.slice(1)} (Task ${r.task_id})</span>
                <span class="sim-result-steps">${r.steps} steps</span>
                <div class="sim-result-bar-wrap">
                    <div class="sim-result-bar" style="width:${pct}%; background:${barColor};"></div>
                </div>
                <span class="sim-result-score ${colorClass}">${score.toFixed(3)}</span>
            </div>
        `;
    });

    // Overall
    const avg = results.length > 0 ? totalScore / results.length : 0;
    const avgPct = Math.round(avg * 100);
    const avgColor = avg >= 0.7 ? 'var(--success)' : avg >= 0.4 ? 'var(--warning)' : 'var(--error)';
    const avgClass = avg >= 0.7 ? 'grade-color-high' : avg >= 0.4 ? 'grade-color-mid' : 'grade-color-low';

    html += `
        <div class="sim-result-row overall">
            <span class="sim-result-name">Overall Average</span>
            <span class="sim-result-steps">${totalSteps} total</span>
            <div class="sim-result-bar-wrap">
                <div class="sim-result-bar" style="width:${avgPct}%; background:${avgColor};"></div>
            </div>
            <span class="sim-result-score ${avgClass}">${avg.toFixed(3)}</span>
        </div>
    `;

    grid.innerHTML = html;
    panel.scrollIntoView({ behavior: 'smooth' });
}
