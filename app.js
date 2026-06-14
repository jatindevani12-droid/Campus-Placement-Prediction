const API_BASE = "/api";

// ── State ────────────────────────────────────────────────────────────
let charts = {};
let modelData = null;

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    await loadInitialData();
    setupPredictForm();
});

// ── Navigation ───────────────────────────────────────────────────────
function showSection(sectionId, element) {
    // Nav UI
    document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');

    // Sections UI
    document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
    document.getElementById(`${sectionId}-section`).classList.add('active');

    // Header updates
    const titleMap = {
        'dashboard': { title: 'Dashboard Overview', sub: 'Real-time statistics and placement patterns' },
        'models': { title: 'ML Performance', sub: 'Comparison of 4 advanced classifiers' },
        'predict': { title: 'Placement Predictor', sub: 'Run inference using our best performing model' }
    };
    
    document.getElementById('section-title').textContent = titleMap[sectionId].title;
    document.getElementById('section-subtitle').textContent = titleMap[sectionId].sub;

    // Refresh charts if needed (some charts might need resize logic if they were hidden)
    Object.values(charts).forEach(c => c.resize());
}

// ── Data Fetching ────────────────────────────────────────────────────
async function loadInitialData() {
    try {
        // Stats
        const statsRes = await fetch(`${API_BASE}/stats`);
        const stats = await statsRes.json();
        updateStatsUI(stats);

        // Charts
        const chartsRes = await fetch(`${API_BASE}/charts`);
        const chartData = await chartsRes.json();
        initOverviewCharts(chartData);

        // Models
        const modelsRes = await fetch(`${API_BASE}/models`);
        modelData = await modelsRes.json();
        initModelCharts(modelData, chartData.feature_importance);

    } catch (err) {
        console.error("Failed to fetch data:", err);
    }
}

function updateStatsUI(stats) {
    document.getElementById('stat-total').textContent = stats.total_students;
    document.getElementById('stat-placed').textContent = stats.placed_count;
    document.getElementById('stat-not-placed').textContent = stats.not_placed_count;
    document.getElementById('stat-rate').textContent = `${stats.placement_rate}%`;
    document.getElementById('total-students').textContent = `Total: ${stats.total_students}`;
    document.getElementById('best-model-name').textContent = stats.best_model.name;
}

// ── Chart Initializations ────────────────────────────────────────────
function initOverviewCharts(data) {
    // 1. Distribution Chart
    const distCtx = document.getElementById('distChart').getContext('2d');
    charts.dist = new Chart(distCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data.placement_distribution),
            datasets: [{
                data: Object.values(data.placement_distribution),
                backgroundColor: Object.keys(data.placement_distribution).map(label => 
                    label === 'Placed' ? '#10b981' : '#ef4444'
                ),
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: { legend: { position: 'bottom' } }
        }
    });

    // 2. Gender Chart
    const genderCtx = document.getElementById('genderChart').getContext('2d');
    const genderLabels = Object.keys(data.gender_distribution);
    charts.gender = new Chart(genderCtx, {
        type: 'bar',
        data: {
            labels: genderLabels,
            datasets: [
                {
                    label: 'Not Placed',
                    data: genderLabels.map(g => data.gender_distribution[g]['Not Placed'] || 0),
                    backgroundColor: '#f87171' // Redder tint
                },
                {
                    label: 'Placed',
                    data: genderLabels.map(g => data.gender_distribution[g]['Placed'] || 0),
                    backgroundColor: '#34d399' // Green tint
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // 3. WorkEx Chart
    const workexCtx = document.getElementById('workexChart').getContext('2d');
    const workexLabels = Object.keys(data.workex_distribution);
    charts.workex = new Chart(workexCtx, {
        type: 'bar',
        data: {
            labels: workexLabels,
            datasets: [
                {
                    label: 'Not Placed',
                    data: workexLabels.map(w => data.workex_distribution[w]['Not Placed'] || 0),
                    backgroundColor: '#f87171'
                },
                {
                    label: 'Placed',
                    data: workexLabels.map(w => data.workex_distribution[w]['Placed'] || 0),
                    backgroundColor: '#34d399'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function initModelCharts(models, importance) {
    // 1. Comparison Radar Chart
    const modelNames = Object.keys(models);
    const metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
    const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'];

    const modelsCtx = document.getElementById('modelsChart').getContext('2d');
    charts.models = new Chart(modelsCtx, {
        type: 'bar',
        data: {
            labels: metricLabels,
            datasets: modelNames.map((name, i) => ({
                label: name,
                data: metrics.map(m => models[name].metrics[m]),
                backgroundColor: `hsla(${i * 60}, 70%, 50%, 0.8)`,
                borderColor: `hsla(${i * 60}, 70%, 50%, 1)`,
                borderWidth: 1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1.0 } },
            plugins: { legend: { position: 'right' } }
        }
    });

    // 2. ROC Curves
    const rocCtx = document.getElementById('rocChart').getContext('2d');
    charts.roc = new Chart(rocCtx, {
        type: 'line',
        data: {
            datasets: modelNames.map((name, i) => ({
                label: name,
                data: models[name].roc_curve.fpr.map((fpr, idx) => ({ x: fpr, y: models[name].roc_curve.tpr[idx] })),
                borderColor: `hsla(${i * 60}, 70%, 50%, 1)`,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: 'linear', position: 'bottom', title: { display: true, text: 'False Positive Rate' } },
                y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' } }
            }
        }
    });

    // 3. Feature Importance
    const impCtx = document.getElementById('importanceChart').getContext('2d');
    charts.importance = new Chart(impCtx, {
        type: 'bar',
        indexAxis: 'y',
        data: {
            labels: importance.map(i => i.feature),
            datasets: [{
                label: 'Importance',
                data: importance.map(i => i.importance),
                backgroundColor: '#6366f1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

// ── Predictor Logic ──────────────────────────────────────────────────
function setupPredictForm() {
    const form = document.getElementById('predict-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('submit-btn');
        btn.disabled = true;
        btn.innerHTML = '<span>Processing...</span>';

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());
        
        // Convert numbers
        ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'].forEach(key => {
            payload[key] = parseFloat(payload[key]);
        });

        try {
            const res = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await res.json();
            displayResult(result);
        } catch (err) {
            console.error("Prediction failed:", err);
            alert("Failed to get prediction from server.");
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<span>Run Prediction</span> <i data-lucide="sparkles"></i>';
            lucide.createIcons();
        }
    });
}

function displayResult(result) {
    const container = document.getElementById('result-container');
    container.classList.remove('hidden');
    container.className = `result-card ${result.prediction === 'Placed' ? 'placed' : 'not-placed'}`;
    
    container.innerHTML = `
        <div class="result-header">
            <h3 class="result-title" style="color: ${result.prediction === 'Placed' ? '#10b981' : '#ef4444'}">
                ${result.prediction}
            </h3>
            <p class="result-prob">Confidence Score: <strong>${result.confidence}%</strong></p>
        </div>
        <div class="result-footer" style="margin-top: 20px; font-size: 0.8rem; color: #64748b;">
            Model: ${result.model_used} | Prob(Placed): ${result.probabilities.Placed}%
        </div>
    `;
}
