/**
 * Attendance Tracker – Frontend Application
 * SPA with hash-based routing, webcam capture, liveness detection,
 * multi-capture registration, and REST API integration.
 */

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
    currentPage: 'dashboard',
    webcamStream: null,
    recognitionInterval: null,
    isRecognizing: false,
    // Multi-capture registration
    registerCaptures: [],
    registerStep: 0,
    // Attendance mode: 'liveness' or 'quick'
    attendanceMode: 'liveness',
    // Liveness
    livenessStream: null,
    livenessFrames: [],
    livenessCapturing: false,
};

const REGISTER_STEPS = [
    { label: 'Look straight at the camera', guide: 'Look Straight' },
    { label: 'Turn your head slightly LEFT', guide: 'Turn Left' },
    { label: 'Turn your head slightly RIGHT', guide: 'Turn Right' },
];

// ─── Router ───────────────────────────────────────────────────────────────────
function navigateTo(page) {
    state.currentPage = page;
    stopWebcam();

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    const pageEl = document.getElementById(`page-${page}`);
    const navEl = document.querySelector(`[data-page="${page}"]`);
    if (pageEl) pageEl.classList.add('active');
    if (navEl) navEl.classList.add('active');

    window.location.hash = page;

    // Reset registration state
    if (page === 'register') resetRegisterState();

    // Load data
    if (page === 'dashboard') loadDashboard();
    if (page === 'records') loadRecords();
    if (page === 'users') loadUsers();
}

window.addEventListener('hashchange', () => {
    const page = window.location.hash.replace('#', '') || 'dashboard';
    navigateTo(page);
});

// ─── Toasts ───────────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiFetch(url, options = {}) {
    try {
        const res = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        return data;
    } catch (err) {
        throw err;
    }
}

// ─── Live Clock ───────────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    const clockEl = document.getElementById('live-clock');
    const dateEl = document.getElementById('live-date');
    if (clockEl) {
        clockEl.textContent = now.toLocaleTimeString('en-IN', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
        });
    }
    if (dateEl) {
        dateEl.textContent = now.toLocaleDateString('en-IN', {
            weekday: 'long', year: 'numeric', month: 'short', day: 'numeric'
        });
    }
}

// ─── Dashboard ────────────────────────────────────────────────────────────────
async function loadDashboard() {
    try {
        const stats = await apiFetch('/api/stats');
        document.getElementById('stat-users').textContent = stats.total_users;
        document.getElementById('stat-today').textContent = stats.today_attendance;
        document.getElementById('stat-total').textContent = stats.total_records;
        document.getElementById('stat-streak').textContent = stats.best_streak + 'd';
        const streakDetail = document.getElementById('stat-streak-name');
        if (streakDetail) {
            streakDetail.textContent = stats.best_streak > 0 ? stats.best_streak_name : '—';
        }
    } catch (err) {
        console.error('Failed to load stats:', err);
    }

    // Load heatmap
    try {
        const data = await apiFetch('/api/heatmap');
        renderHeatmap(data.heatmap);
    } catch (err) {
        console.error('Failed to load heatmap:', err);
    }
}

function renderHeatmap(heatmap) {
    const container = document.getElementById('weekly-heatmap');
    if (!container || !heatmap) return;

    const maxCount = Math.max(...heatmap.map(d => d.count), 1);

    container.innerHTML = heatmap.map(d => {
        const intensity = d.count / maxCount;
        const bgColor = d.count === 0
            ? 'rgba(255,255,255,0.04)'
            : `rgba(16, 185, 129, ${0.15 + intensity * 0.6})`;
        const todayClass = d.is_today ? 'heatmap-today' : '';
        return `
            <div class="heatmap-cell ${todayClass}" style="background: ${bgColor}">
                <div class="heatmap-day">${d.day}</div>
                <div class="heatmap-count">${d.count}</div>
                <div class="heatmap-date-label">${d.date.slice(5)}</div>
            </div>
        `;
    }).join('');
}

// ─── Webcam ───────────────────────────────────────────────────────────────────
async function startWebcam(videoId) {
    try {
        const video = document.getElementById(videoId);
        if (!video) return;
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        video.srcObject = stream;
        state.webcamStream = stream;
        return true;
    } catch (err) {
        showToast('Camera access denied. Please allow camera permissions.', 'error');
        return false;
    }
}

function stopWebcam() {
    if (state.recognitionInterval) {
        clearInterval(state.recognitionInterval);
        state.recognitionInterval = null;
    }
    state.isRecognizing = false;
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(t => t.stop());
        state.webcamStream = null;
    }
}

function captureFrame(videoId) {
    const video = document.getElementById(videoId);
    if (!video || !video.srcObject) return null;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
}

// ─── Multi-Capture Registration ───────────────────────────────────────────────
function resetRegisterState() {
    state.registerCaptures = [];
    state.registerStep = 0;
    updateRegisterUI();
}

function updateRegisterUI() {
    const step = state.registerStep;
    // Update progress steps
    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById(`step-${i}`);
        if (!el) continue;
        el.classList.remove('active', 'done');
        if (i - 1 < step) el.classList.add('done');
        if (i - 1 === step && step < 3) el.classList.add('active');
    }
    // Update guide text
    const guide = document.getElementById('register-guide');
    const guideText = document.getElementById('register-guide-text');
    if (guide && guideText && step < 3) {
        guideText.textContent = REGISTER_STEPS[step].label;
    }
    // Update button
    const btn = document.getElementById('btn-capture');
    if (btn) {
        if (step >= 3) {
            btn.innerHTML = '<span class="spinner"></span> Registering…';
            btn.disabled = true;
        } else {
            btn.innerHTML = `📸 Capture (Step ${step + 1}/3)`;
        }
    }
    // Info
    const info = document.getElementById('register-captures-info');
    const countEl = document.getElementById('captures-count');
    if (info && countEl) {
        if (step > 0) {
            info.style.display = 'block';
            countEl.textContent = `${step}/3 captured`;
        } else {
            info.style.display = 'none';
        }
    }
}

async function startRegisterCamera() {
    const started = await startWebcam('register-video');
    if (started) {
        document.getElementById('register-placeholder').style.display = 'none';
        document.getElementById('register-video').style.display = 'block';
        document.getElementById('register-guide').style.display = 'flex';
        document.getElementById('btn-start-register-cam').style.display = 'none';
        document.getElementById('btn-capture').disabled = false;
        resetRegisterState();
    }
}

async function captureMultiStep() {
    const name = document.getElementById('register-name').value.trim();
    if (!name) {
        showToast('Please enter a name first.', 'error');
        return;
    }

    const frame = captureFrame('register-video');
    if (!frame) {
        showToast('No webcam frame available.', 'error');
        return;
    }

    state.registerCaptures.push(frame);
    state.registerStep++;

    showToast(`Capture ${state.registerStep}/3 ✓`, 'success');
    updateRegisterUI();

    if (state.registerStep >= 3) {
        // Submit all 3
        try {
            const result = await apiFetch('/api/register-multi', {
                method: 'POST',
                body: JSON.stringify({ name, images: state.registerCaptures }),
            });
            showToast(result.message, 'success');
            document.getElementById('register-name').value = '';
            resetRegisterState();
        } catch (err) {
            showToast(err.message, 'error');
            resetRegisterState();
        }
    }
}

async function captureSingleRegister() {
    const name = document.getElementById('register-name').value.trim();
    if (!name) {
        showToast('Please enter a name first.', 'error');
        return;
    }
    const frame = captureFrame('register-video');
    if (!frame) {
        showToast('Start the camera first.', 'error');
        return;
    }
    const btn = document.getElementById('btn-single-register');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing…';
    try {
        const result = await apiFetch('/api/register', {
            method: 'POST',
            body: JSON.stringify({ name, image: frame }),
        });
        showToast(result.message, 'success');
        document.getElementById('register-name').value = '';
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Quick Register (1 shot)';
    }
}

// ─── Attendance Mode ──────────────────────────────────────────────────────────
function setAttendanceMode(mode) {
    state.attendanceMode = mode;
    document.getElementById('mode-liveness').classList.toggle('active', mode === 'liveness');
    document.getElementById('mode-quick').classList.toggle('active', mode === 'quick');
}

function startAttendance() {
    if (state.attendanceMode === 'liveness') {
        openLivenessModal();
    } else {
        startRecognition();
    }
}

// ─── Quick Recognition (no liveness) ─────────────────────────────────────────
async function startRecognition() {
    const started = await startWebcam('recognize-video');
    if (!started) return;

    document.getElementById('recognize-placeholder').style.display = 'none';
    document.getElementById('recognize-video').style.display = 'block';
    document.getElementById('recognize-canvas').style.display = 'block';
    document.getElementById('btn-start-recognize').style.display = 'none';
    document.getElementById('btn-stop-recognize').style.display = 'inline-flex';
    document.getElementById('recognition-status').style.display = 'flex';
    document.getElementById('recognition-status-text').textContent = 'Scanning faces (Quick Mode — no liveness)…';

    state.isRecognizing = true;

    state.recognitionInterval = setInterval(async () => {
        if (!state.isRecognizing) return;

        const frame = captureFrame('recognize-video');
        if (!frame) return;

        try {
            const result = await apiFetch('/api/recognize', {
                method: 'POST',
                body: JSON.stringify({ image: frame }),
            });
            drawFaceBoxes(result);
            updateRecognizedList(result.recognized);
        } catch (err) {
            console.error('Recognition error:', err);
        }
    }, 2000);
}

function stopRecognition() {
    stopWebcam();
    document.getElementById('recognize-video').style.display = 'none';
    document.getElementById('recognize-canvas').style.display = 'none';
    document.getElementById('recognize-placeholder').style.display = 'flex';
    document.getElementById('btn-start-recognize').style.display = 'inline-flex';
    document.getElementById('btn-stop-recognize').style.display = 'none';
    document.getElementById('recognition-status').style.display = 'none';

    const canvas = document.getElementById('recognize-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawFaceBoxes(result) {
    const video = document.getElementById('recognize-video');
    const canvas = document.getElementById('recognize-canvas');
    if (!video || !canvas) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result.recognized) return;

    result.recognized.forEach(person => {
        const [top, right, bottom, left] = person.location;
        const w = right - left;
        const h = bottom - top;

        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(left, top, w, h);

        ctx.fillStyle = 'rgba(16, 185, 129, 0.85)';
        const textWidth = ctx.measureText(person.name).width;
        ctx.fillRect(left, top - 26, textWidth + 16, 24);

        ctx.fillStyle = 'white';
        ctx.font = '600 13px Inter, sans-serif';
        ctx.fillText(person.name, left + 8, top - 8);
    });
}

function updateRecognizedList(recognized) {
    const container = document.getElementById('recognized-list');
    if (!recognized || recognized.length === 0) return;

    recognized.forEach(person => {
        if (container.querySelector(`[data-uid="${person.user_id}"]`)) return;

        const item = document.createElement('div');
        item.className = 'recognized-item';
        item.setAttribute('data-uid', person.user_id);

        let badgeText = 'Recorded';
        let badgeClass = 'badge-success';
        if (person.holiday_blocked) {
            badgeText = 'Blocked (Holiday)';
            badgeClass = 'badge-warning';
        } else if (person.already_recorded) {
            badgeText = 'Already Recorded';
        }

        item.innerHTML = `
            <span style="font-size:1.2rem">👤</span>
            <span class="ri-name">${person.name}</span>
            <span class="ri-conf">${Math.round(person.confidence * 100)}% match</span>
            <span class="badge ${badgeClass}">${badgeText}</span>
        `;
        container.prepend(item);

        if (person.holiday_blocked) {
            showToast(`Attendance not marked for ${person.name} (Sunday/Holiday)`, 'warning');
        } else if (!person.already_recorded) {
            showToast(`Attendance recorded for ${person.name}`, 'success');
        }
    });
}

// ─── Liveness Challenge ───────────────────────────────────────────────────────
async function openLivenessModal() {
    const modal = document.getElementById('liveness-modal');
    modal.style.display = 'flex';

    // Reset UI
    document.getElementById('liveness-instruction').textContent = 'Preparing camera…';
    document.getElementById('liveness-result').style.display = 'none';
    document.getElementById('btn-start-liveness').style.display = 'inline-flex';
    document.getElementById('btn-start-liveness').disabled = false;
    document.getElementById('step-blink').className = 'liveness-step';
    document.getElementById('step-texture').className = 'liveness-step';
    resetLivenessRing();

    // Start liveness camera
    try {
        const video = document.getElementById('liveness-video');
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        video.srcObject = stream;
        state.livenessStream = stream;
        document.getElementById('liveness-instruction').textContent = 'Camera ready. Press Start to begin the liveness check.';
    } catch (err) {
        document.getElementById('liveness-instruction').textContent = 'Camera access denied.';
        showToast('Camera access denied for liveness check.', 'error');
    }
}

function cancelLiveness() {
    if (state.livenessStream) {
        state.livenessStream.getTracks().forEach(t => t.stop());
        state.livenessStream = null;
    }
    state.livenessCapturing = false;
    document.getElementById('liveness-modal').style.display = 'none';
}

function resetLivenessRing() {
    const ring = document.getElementById('liveness-ring-progress');
    if (ring) {
        ring.style.strokeDashoffset = '339.292';
    }
}

function animateLivenessRing(progress) {
    const ring = document.getElementById('liveness-ring-progress');
    if (ring) {
        const circumference = 2 * Math.PI * 54;
        ring.style.strokeDashoffset = circumference * (1 - progress);
    }
}

async function runLivenessCheck() {
    const btn = document.getElementById('btn-start-liveness');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Checking…';

    const instruction = document.getElementById('liveness-instruction');
    instruction.textContent = '👁️ Please BLINK your eyes naturally…';
    instruction.classList.add('blink-prompt');

    state.livenessFrames = [];
    state.livenessCapturing = true;

    // Capture frames over 3 seconds
    const totalDuration = 3000;
    const captureInterval = 200; // every 200ms
    const totalFrames = totalDuration / captureInterval;
    let framesCaptured = 0;

    await new Promise(resolve => {
        const interval = setInterval(() => {
            if (!state.livenessCapturing) {
                clearInterval(interval);
                resolve();
                return;
            }

            const video = document.getElementById('liveness-video');
            if (video && video.srcObject) {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                state.livenessFrames.push(canvas.toDataURL('image/jpeg', 0.8));
            }
            framesCaptured++;
            animateLivenessRing(framesCaptured / totalFrames);

            if (framesCaptured >= totalFrames) {
                clearInterval(interval);
                resolve();
            }
        }, captureInterval);
    });

    instruction.textContent = '🔍 Analyzing frames…';
    instruction.classList.remove('blink-prompt');

    // Send to backend
    try {
        const result = await apiFetch('/api/recognize-with-liveness', {
            method: 'POST',
            body: JSON.stringify({ frames: state.livenessFrames }),
        });

        // Update step indicators
        document.getElementById('step-blink').classList.add(result.liveness ? 'pass' : 'fail');
        document.getElementById('step-texture').classList.add(result.liveness ? 'pass' : 'fail');

        const resultDiv = document.getElementById('liveness-result');
        resultDiv.style.display = 'block';

        if (result.liveness && result.recognized && result.recognized.length > 0) {
            resultDiv.className = 'liveness-result success';
            const isHoliday = result.recognized[0].holiday_blocked;
            resultDiv.innerHTML = `
                <div class="result-icon">${isHoliday ? 'ℹ️' : '✓'}</div>
                <div class="result-text">
                    <strong>${result.message}</strong>
                    <span>Confidence: ${Math.round(result.recognized[0].confidence * 100)}%</span>
                </div>
            `;
            showToast(result.message, isHoliday ? 'warning' : 'success');
            // Auto close after 2s
            setTimeout(() => cancelLiveness(), 2500);
        } else {
            resultDiv.className = 'liveness-result fail';
            resultDiv.innerHTML = `
                <div class="result-icon">✕</div>
                <div class="result-text">
                    <strong>${result.message}</strong>
                    <span>Please try again</span>
                </div>
            `;
            showToast(result.message, 'error');
        }

    } catch (err) {
        showToast('Liveness check failed: ' + err.message, 'error');
    }

    btn.disabled = false;
    btn.innerHTML = '🔄 Retry';
}

// ─── Records ──────────────────────────────────────────────────────────────────
async function loadRecords() {
    const dateInput = document.getElementById('filter-date');
    const dateVal = dateInput ? dateInput.value : '';
    const url = dateVal ? `/api/attendance?date=${dateVal}` : '/api/attendance';

    try {
        const data = await apiFetch(url);
        renderRecordsTable(data.records);
    } catch (err) {
        showToast('Failed to load records', 'error');
    }
}

function renderRecordsTable(records) {
    const tbody = document.getElementById('records-tbody');
    const empty = document.getElementById('records-empty');

    if (!records.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    const emotionEmojis = {
        'Happy': '😊',
        'Neutral': '😐',
        'Sad': '😟',
        'Angry': '😡',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢'
    };

    tbody.innerHTML = records.map((r, i) => {
        const lateBadge = r.late_status === 'Late'
            ? '<span class="badge badge-warning">⏰ Late</span>'
            : '<span class="badge badge-success">● On Time</span>';
        const verifiedBadge = r.liveness_verified
            ? '<span class="badge badge-verified">🛡️ Verified</span>'
            : '<span class="badge badge-unverified">— Quick</span>';

        return `
        <tr>
            <td>${i + 1}</td>
            <td style="color: var(--text-primary); font-weight: 500">${r.name}</td>
            <td>${new Date(r.timestamp).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' })}</td>
            <td>${new Date(r.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</td>
            <td><span class="badge badge-success">● ${r.status}</span></td>
            <td>${lateBadge}</td>
            <td>${verifiedBadge}</td>
        </tr>
    `;
    }).join('');
}

function exportCSV() {
    const dateInput = document.getElementById('filter-date');
    const dateVal = dateInput ? dateInput.value : '';
    const url = dateVal ? `/api/export-csv?date=${dateVal}` : '/api/export-csv';
    window.open(url, '_blank');
    showToast('CSV download started', 'info');
}

// ─── Users ────────────────────────────────────────────────────────────────────
async function loadUsers() {
    try {
        const data = await apiFetch('/api/users');
        renderUsers(data.users);
    } catch (err) {
        showToast('Failed to load users', 'error');
    }
}

function renderUsers(users) {
    const container = document.getElementById('users-grid');
    const empty = document.getElementById('users-empty');

    if (!users.length) {
        container.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    container.innerHTML = users.map(u => `
        <div class="user-card">
            <div class="user-avatar">${u.name.charAt(0).toUpperCase()}</div>
            <div class="user-info">
                <div class="name">${u.name}</div>
                <div class="date">Registered ${new Date(u.created_at).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' })}</div>
            </div>
            <button class="delete-btn" onclick="deleteUser(${u.id}, '${u.name}')" title="Delete user">🗑</button>
        </div>
    `).join('');
}

async function deleteUser(id, name) {
    if (!confirm(`Delete "${name}"? This will also remove their attendance records.`)) return;
    try {
        await apiFetch(`/api/users/${id}`, { method: 'DELETE' });
        showToast(`${name} deleted`, 'info');
        loadUsers();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ─── Mobile sidebar toggle ───────────────────────────────────────────────────
function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('open');
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('data-page');
            navigateTo(page);
            document.querySelector('.sidebar').classList.remove('open');
        });
    });

    // Initial page from hash
    const initial = window.location.hash.replace('#', '') || 'dashboard';
    navigateTo(initial);

    // Start live clock
    updateClock();
    setInterval(updateClock, 1000);
});
