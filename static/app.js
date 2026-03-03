/**
 * Attendance Tracker – Frontend Application
 * SPA with hash-based routing, webcam capture, and REST API integration.
 */

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
    currentPage: 'dashboard',
    webcamStream: null,
    recognitionInterval: null,
    isRecognizing: false,
};

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

    // Load data for the page
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
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
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

// ─── Dashboard ────────────────────────────────────────────────────────────────
async function loadDashboard() {
    try {
        const stats = await apiFetch('/api/stats');
        document.getElementById('stat-users').textContent = stats.total_users;
        document.getElementById('stat-today').textContent = stats.today_attendance;
        document.getElementById('stat-total').textContent = stats.total_records;
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
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

// ─── Registration ─────────────────────────────────────────────────────────────
async function startRegisterCamera() {
    const started = await startWebcam('register-video');
    if (started) {
        document.getElementById('register-placeholder').style.display = 'none';
        document.getElementById('register-video').style.display = 'block';
        document.getElementById('btn-start-register-cam').style.display = 'none';
        document.getElementById('btn-capture').disabled = false;
    }
}

async function captureAndRegister() {
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

    const btn = document.getElementById('btn-capture');
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
        btn.innerHTML = '📸 Capture & Register';
    }
}

// ─── Attendance Recognition ───────────────────────────────────────────────────
async function startRecognition() {
    const started = await startWebcam('recognize-video');
    if (!started) return;

    document.getElementById('recognize-placeholder').style.display = 'none';
    document.getElementById('recognize-video').style.display = 'block';
    document.getElementById('recognize-canvas').style.display = 'block';
    document.getElementById('btn-start-recognize').style.display = 'none';
    document.getElementById('btn-stop-recognize').style.display = 'inline-flex';
    document.getElementById('recognition-status').style.display = 'flex';

    state.isRecognizing = true;

    // Send frames every 2 seconds
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

        // Name label
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
        // Avoid adding duplicates already on screen
        if (container.querySelector(`[data-uid="${person.user_id}"]`)) return;

        const item = document.createElement('div');
        item.className = 'recognized-item';
        item.setAttribute('data-uid', person.user_id);
        item.innerHTML = `
            <span style="font-size:1.2rem">✓</span>
            <span class="ri-name">${person.name}</span>
            <span class="ri-conf">${Math.round(person.confidence * 100)}% match</span>
            <span class="badge badge-success">${person.already_recorded ? 'Already Recorded' : 'Recorded'}</span>
        `;
        container.prepend(item);

        if (!person.already_recorded) {
            showToast(`Attendance recorded for ${person.name}`, 'success');
        }
    });
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

    tbody.innerHTML = records.map((r, i) => `
        <tr>
            <td>${i + 1}</td>
            <td style="color: var(--text-primary); font-weight: 500">${r.name}</td>
            <td>${new Date(r.timestamp).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' })}</td>
            <td>${new Date(r.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</td>
            <td><span class="badge badge-success">● ${r.status}</span></td>
        </tr>
    `).join('');
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
            // Close sidebar on mobile
            document.querySelector('.sidebar').classList.remove('open');
        });
    });

    // Initial page from hash
    const initial = window.location.hash.replace('#', '') || 'dashboard';
    navigateTo(initial);
});
