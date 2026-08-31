// DiagnoseHub — shared interactivity across screens
// Each function guards on element existence so this one file can be
// safely included on every page.

const DIAGNOSEHUB_STORAGE_KEY = 'diagnosehub_profile';
const DIAGNOSEHUB_RESULTS_KEY = 'diagnosehub_results';
const API_BASE_URL = 'http://localhost:5000';

document.addEventListener("DOMContentLoaded", function () {
  initStepper();
  initSegmented();
  initYesNoQuestions();
  initOptionLists();
  initEmojiScale();
  initBmiCalculator();
  initWizardFlow();
  initLoadingRedirect();
  initSaveToast();
  initResultPage();
});

/* ---------- Age stepper (step1) ---------- */
function initStepper() {
  var value = document.getElementById("usia-value");
  if (!value) return;

  var min = 1, max = 120;

  document.querySelectorAll(".stepper-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var current = parseInt(value.textContent, 10) || 0;
      if (btn.dataset.action === "inc") current = Math.min(max, current + 1);
      if (btn.dataset.action === "dec") current = Math.max(min, current - 1);
      value.textContent = current;
    });
  });
}

/* ---------- Generic 2-way segmented toggle (gender) ---------- */
function initSegmented() {
  document.querySelectorAll(".segmented--gender").forEach(function (group) {
    var options = group.querySelectorAll(".segmented-option");
    options.forEach(function (opt) {
      opt.addEventListener("click", function () {
        options.forEach(function (o) { o.classList.remove("is-active"); });
        opt.classList.add("is-active");
      });
    });
  });
}

/* ---------- Ya / Tidak questions (step2) ---------- */
function initYesNoQuestions() {
  document.querySelectorAll(".segmented--yesno").forEach(function (group) {
    var options = group.querySelectorAll(".segmented-option");
    options.forEach(function (opt) {
      opt.addEventListener("click", function () {
        options.forEach(function (o) {
          o.classList.remove("is-active", "is-active-danger", "is-active-success");
        });
        if (opt.classList.contains("option-yes")) {
          opt.classList.add("is-active-danger");
        } else {
          opt.classList.add("is-active-success");
        }
      });
    });
  });
}

/* ---------- Option lists (status merokok / aktivitas fisik) ---------- */
function initOptionLists() {
  document.querySelectorAll(".option-list").forEach(function (list) {
    var items = list.querySelectorAll(".option-item");
    items.forEach(function (item) {
      item.addEventListener("click", function () {
        items.forEach(function (i) { i.classList.remove("is-active"); });
        item.classList.add("is-active");
      });
    });
  });
}

/* ---------- Emoji scale (kondisi kesehatan umum) ---------- */
function initEmojiScale() {
  var scale = document.querySelector(".emoji-scale");
  if (!scale) return;
  var options = scale.querySelectorAll(".emoji-option");
  options.forEach(function (opt) {
    opt.addEventListener("click", function () {
      options.forEach(function (o) { o.classList.remove("is-active"); });
      opt.classList.add("is-active");
    });
  });
}

/* ---------- BMI calculator (step1) ---------- */
function initBmiCalculator() {
  var beratInput = document.getElementById("berat");
  var tinggiInput = document.getElementById("tinggi");
  if (!beratInput || !tinggiInput) return;

  var card = document.getElementById("bmi-card");
  var circle = document.getElementById("bmi-circle");
  var label = document.getElementById("bmi-label");
  var tag = document.getElementById("bmi-tag");
  var desc = document.getElementById("bmi-desc");

  function update() {
    var berat = parseFloat(beratInput.value);
    var tinggiCm = parseFloat(tinggiInput.value);
    if (!berat || !tinggiCm) return;

    var tinggiM = tinggiCm / 100;
    var bmi = berat / (tinggiM * tinggiM);
    var bmiRounded = Math.round(bmi * 10) / 10;

    circle.textContent = bmiRounded.toFixed(1);

    card.classList.remove("is-normal", "is-danger");
    tag.classList.remove("tag--success", "tag--warning", "tag--danger");

    if (bmi < 18.5) {
      label.textContent = "Kurang (Underweight)";
      tag.textContent = "RENDAH";
      tag.classList.add("tag--warning");
      desc.textContent = "Di bawah rentang ideal. Tambah asupan gizi seimbang.";
    } else if (bmi < 25) {
      label.textContent = "Ideal (Normal)";
      tag.textContent = "BAIK";
      tag.classList.add("tag--success");
      card.classList.add("is-normal");
      desc.textContent = "Berat badan Anda berada pada rentang ideal. Pertahankan.";
    } else if (bmi < 30) {
      label.textContent = "Berlebih (Overweight)";
      tag.textContent = "SEDANG";
      tag.classList.add("tag--warning");
      desc.textContent = "Sedikit di atas rentang ideal. Jaga pola makan sehat.";
    } else {
      label.textContent = "Obesitas (Obese)";
      tag.textContent = "TINGGI";
      tag.classList.add("tag--danger");
      card.classList.add("is-danger");
      desc.textContent = "Jauh di atas rentang ideal. Disarankan konsultasi ke Puskesmas.";
    }
  }

  beratInput.addEventListener("input", update);
  tinggiInput.addEventListener("input", update);
  update();
}

/* ---------- Assessment wizard state + API flow ---------- */
function getStoredAssessment() {
  try {
    return JSON.parse(localStorage.getItem(DIAGNOSEHUB_STORAGE_KEY)) || {};
  } catch (error) {
    return {};
  }
}

function saveStoredAssessment(data) {
  localStorage.setItem(DIAGNOSEHUB_STORAGE_KEY, JSON.stringify(data));
}

function getCurrentPageName() {
  return window.location.pathname.split('/').pop() || 'index.html';
}

function collectStep1Data() {
  var age = parseInt(document.getElementById('usia-value')?.textContent || '45', 10) || 45;
  var gender = document.querySelector('.segmented--gender .is-active')?.dataset.value || 'laki-laki';
  var berat = parseFloat(document.getElementById('berat')?.value || '70');
  var tinggi = parseFloat(document.getElementById('tinggi')?.value || '165');
  var bmi = berat && tinggi ? berat / ((tinggi / 100) * (tinggi / 100)) : 25;

  return {
    age: age,
    gender: gender,
    berat: berat,
    tinggi: tinggi,
    bmi: Number(bmi.toFixed(1))
  };
}

function collectStep2Data() {
  var questions = document.querySelectorAll('.question-card');
  var result = {};

  questions.forEach(function (card, index) {
    var selected = card.querySelector('.segmented-option.is-active');
    var value = selected ? selected.dataset.value : 'tidak';
    if (index === 0) result.highBP = value === 'ya' ? 1 : 0;
    if (index === 1) result.diabetesHistory = value === 'ya' ? 1 : 0;
    if (index === 2) result.heartAttackHistory = value === 'ya' ? 1 : 0;
    if (index === 3) result.familyHistory = value === 'ya' ? 1 : 0;
  });

  return result;
}

function collectStep3Data() {
  var smokingList = document.querySelectorAll('.option-list')[0];
  var activityList = document.querySelectorAll('.option-list')[1];
  var smoking = smokingList ? smokingList.querySelector('.option-item.is-active')?.dataset.value || 'tidak-merokok' : 'tidak-merokok';
  var activity = activityList ? activityList.querySelector('.option-item.is-active')?.dataset.value || 'kadang' : 'kadang';
  var general = document.querySelector('.emoji-option.is-active')?.dataset.value || 'baik';

  return {
    smoker: smoking === 'masih-merokok' || smoking === 'pernah-merokok' ? 1 : 0,
    alcohol: smoking === 'masih-merokok' ? 1 : 0,
    activity: activity,
    physicalActivity: activity === 'aktif' ? 1 : activity === 'kadang' ? 0.5 : 0,
    generalHealth: general,
    genHlth: mapGeneralHealth(general)
  };
}

function mapGeneralHealth(value) {
  var map = { buruk: 5, cukup: 3, baik: 2, 'sangat-baik': 1, prima: 1 };
  return map[value] || 2;
}

function persistCurrentStep() {
  var existing = getStoredAssessment();
  var page = getCurrentPageName();

  if (page.indexOf('step1') !== -1) {
    Object.assign(existing, collectStep1Data());
  }

  if (page.indexOf('step2') !== -1) {
    Object.assign(existing, collectStep2Data());
  }

  if (page.indexOf('step3') !== -1) {
    Object.assign(existing, collectStep3Data());
  }

  saveStoredAssessment(existing);
}

function buildPredictionPayload() {
  var survey = getStoredAssessment();
  var age = Number(survey.age || 45);
  var sex = survey.gender === 'perempuan' ? 0 : 1;
  var bmi = Number(survey.bmi || 25);
  var highBP = Number(survey.highBP || 0);
  var heartDisease = Number(survey.heartAttackHistory || survey.familyHistory || 0);
  var smoker = Number(survey.smoker || 0);
  var physicalActivity = Number(survey.physicalActivity || 0);
  var genHlth = Number(survey.genHlth || 2);
  var highChol = bmi > 25 || genHlth >= 3 ? 1 : 0;
  var fruits = 1;
  var veggies = 1;

  return {
    diabetes: {
      HighBP: highBP,
      HighChol: highChol,
      CholCheck: 1,
      BMI: bmi,
      Smoker: smoker,
      Stroke: Number(survey.heartAttackHistory || 0),
      HeartDiseaseorAttack: heartDisease,
      PhysActivity: physicalActivity,
      Fruits: fruits,
      Veggies: veggies,
      HvyAlcoholConsump: Number(survey.alcohol || 0),
      AnyHealthcare: 1,
      NoDocbcCost: 0,
      GenHlth: genHlth,
      MentHlth: Math.max(0, 5 - genHlth),
      PhysHlth: Math.max(0, genHlth === 5 ? 12 : 5),
      DiffWalk: Number(survey.familyHistory || 0),
      Sex: sex,
      Age: age,
      Education: 4,
      Income: 5
    },
    hypertension: {
      age: age,
      sex: sex,
      cp: heartDisease ? 2 : 0,
      trestbps: highBP ? 145 : 120,
      chol: highChol ? 240 : 180,
      fbs: 0,
      restecg: 0,
      thalach: Math.max(110, 170 - (genHlth * 6)),
      exang: heartDisease ? 1 : 0,
      oldpeak: Number((bmi / 18).toFixed(1)),
      slope: highBP ? 2 : 1,
      ca: 0,
      thal: 2
    },
    oral_cancer: {
      Country: 0,
      Age: age,
      Gender: sex,
      'Tobacco Use': smoker,
      'Alcohol Consumption': Number(survey.alcohol || 0),
      'HPV Infection': 0,
      'Betel Quid Use': 0,
      'Chronic Sun Exposure': 0,
      'Poor Oral Hygiene': 0,
      'Diet (Fruits & Vegetables Intake)': fruits,
      'Family History of Cancer': Number(survey.familyHistory || 0),
      'Compromised Immune System': 0,
      'Oral Lesions': Number(survey.heartAttackHistory || 0),
      'Unexplained Bleeding': Number(survey.heartAttackHistory || 0),
      'Difficulty Swallowing': Number(survey.heartAttackHistory || 0),
      'White or Red Patches in Mouth': Number(survey.heartAttackHistory || 0),
      'Tumor Size (cm)': 0,
      'Cancer Stage': 0,
      'Treatment Type': 0,
      'Survival Rate (5-Year, %)': 0,
      'Cost of Treatment (USD)': 0,
      'Economic Burden (Lost Workdays per Year)': 0,
      'Early Diagnosis': 1
    }
  };
}

function initWizardFlow() {
  document.querySelectorAll('a[href="step2.html"], a[href="step3.html"], a[href="loading.html"]').forEach(function (link) {
    link.addEventListener('click', function (event) {
      var href = link.getAttribute('href');
      if (!href) return;

      if (href === 'loading.html') {
        event.preventDefault();
        persistCurrentStep();
        runRiskAssessment();
        return;
      }

      event.preventDefault();
      persistCurrentStep();
      window.location.href = href;
    });
  });
}

async function runRiskAssessment() {
  var screen = document.querySelector('.loading-screen');
  if (screen) {
    var label = screen.querySelector('h2');
    if (label) label.textContent = 'Menghubungkan ke Model AI';
  }

  var payloads = buildPredictionPayload();
  var requests = [
    fetch(API_BASE_URL + '/predict/diabetes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloads.diabetes)
    }),
    fetch(API_BASE_URL + '/predict/hypertension', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloads.hypertension)
    }),
    fetch(API_BASE_URL + '/predict/oral_cancer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloads.oral_cancer)
    })
  ];

  try {
    var responses = await Promise.all(requests);
    var results = {
      diabetes: await responses[0].json(),
      hypertension: await responses[1].json(),
      oral_cancer: await responses[2].json()
    };

    localStorage.setItem(DIAGNOSEHUB_RESULTS_KEY, JSON.stringify(results));

    if (screen) {
      setTimeout(function () {
        window.location.href = screen.dataset.redirect || 'hasil.html';
      }, 700);
      return;
    }

    window.location.href = 'hasil.html';
  } catch (error) {
    console.error('API request failed:', error);
    if (screen) {
      var message = screen.querySelector('p');
      if (message) message.textContent = 'Gagal terhubung ke server API. Pastikan app.py sedang berjalan di localhost:5000.';
    }
    localStorage.setItem(DIAGNOSEHUB_RESULTS_KEY, JSON.stringify({
      diabetes: { success: false, error: 'API offline' },
      hypertension: { success: false, error: 'API offline' },
      oral_cancer: { success: false, error: 'API offline' }
    }));
  }
}

function initLoadingRedirect() {
  var screen = document.querySelector('.loading-screen');
  if (!screen) return;

  var target = screen.dataset.redirect || 'hasil.html';
  if (window.location.pathname.toLowerCase().endsWith('loading.html')) {
    runRiskAssessment();
    return;
  }

  setTimeout(function () {
    window.location.href = target;
  }, 2600);
}

/* ---------- Results page rendering ---------- */
function initResultPage() {
  var body = document.querySelector('.results-body');
  if (!body) return;

  try {
    var results = JSON.parse(localStorage.getItem(DIAGNOSEHUB_RESULTS_KEY) || '{}');
    renderResultCards(body, results);
  } catch (error) {
    renderResultCards(body, {});
  }
}

function renderResultCards(container, results) {
  var items = [
    { key: 'diabetes', label: 'Diabetes Melitus', icon: '📈', short: 'Diabetes' },
    { key: 'hypertension', label: 'Hipertensi', icon: '❤️', short: 'Hipertensi' },
    { key: 'oral_cancer', label: 'Kanker Mulut', icon: '🦷', short: 'Oral Cancer' }
  ];

  var cardsHtml = items.map(function (item) {
    var result = results[item.key] || {};
    var success = result.success === true;
    var confidence = Number(result.confidence || 0) * 100 || 0;
    var riskLevel = result.risk_level || (confidence > 80 ? 'High' : confidence > 60 ? 'Medium' : 'Low');
    var tagClass = riskLevel === 'High' ? 'tag tag--danger' : riskLevel === 'Medium' ? 'tag tag--warning' : 'tag tag--success';
    var cardClass = riskLevel === 'High' ? 'risk-card risk-card--primary risk-card--tinggi' : riskLevel === 'Medium' ? 'risk-card risk-card--sedang' : 'risk-card risk-card--rendah';
    var riskText = success ? (riskLevel === 'High' ? 'TINGGI' : riskLevel === 'Medium' ? 'SEDANG' : 'RENDAH') : 'BELUM TERSEDIA';
    var predictionText = success ? String(result.prediction || 'Tdk terdeteksi') : 'Data belum tersedia';
    var description = success ? ('Model menilai risiko ' + predictionText + ' dengan keyakinan ' + confidence.toFixed(0) + '%') : 'Tidak dapat diproses dari API';
    var recommendationText = success && result.recommendations && result.recommendations.immediate_actions ? result.recommendations.immediate_actions[0] : 'Periksa kembali koneksi API server';

    return '<article class="' + cardClass + '">'
      + '<div class="risk-card-head">'
      + '<div class="risk-card-title">'
      + '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M20.8 8.6c0 4.7-4.6 8.4-8.8 11.6-4.2-3.2-8.8-6.9-8.8-11.6a5 5 0 0 1 9-3 5 5 0 0 1 8.6 3Z"/></svg>'
      + item.label
      + '</div>'
      + '<div class="risk-card-badges"><span class="' + tagClass + '">' + riskText + '</span><span class="risk-percent">' + confidence.toFixed(0) + '%</span></div>'
      + '</div>'
      + '<p class="risk-desc">' + description + '</p>'
      + '<div class="suggestion-box"><strong>Saran Tindakan:</strong><ul><li>' + recommendationText + '</li></ul></div>'
      + '</article>';
  }).join('');

  container.innerHTML = '<div class="result-summary-card">'
    + '<div class="result-summary-head"><h2>Hasil Prediksi</h2><span class="tag tag--success">SELESAI</span></div>'
    + '<p class="result-date">Diperiksa pada: ' + new Date().toLocaleString('id-ID', { day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' }) + ' WIB</p>'
    + '</div>'
    + '<p class="result-section-label">PREDIKSI LAINNYA</p>'
    + cardsHtml;
}

/* ---------- "Simpan Hasil" toast feedback ---------- */
function initSaveToast() {
  var btn = document.getElementById("simpan-btn");
  var toast = document.getElementById("toast");
  if (!btn || !toast) return;

  btn.addEventListener("click", function () {
    toast.classList.add("is-visible");
    clearTimeout(btn._toastTimer);
    btn._toastTimer = setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 2200);
  });
}
