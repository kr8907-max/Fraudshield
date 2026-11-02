/* ============================================================================
   FraudShield — dashboard frontend
   Preserves the existing FastAPI integration:
     - POST /api/scan  { url }  -> full layered verdict
     - GET  /api/health         -> server-status chip
   No mock data: every rendered scan value comes from the real response.
============================================================================ */

/* ---------------------------------------------------------------- elements */
const scanForm = document.querySelector("#scan-form");
const urlInput = document.querySelector("#url-input");
const scanButton = document.querySelector("#scan-button");
const scanMessage = document.querySelector("#scan-message");

const scanProgress = document.querySelector("#scan-progress");
const progressStagesList = document.querySelector("#progress-stages");
const progressTimerDisplayElement = document.querySelector("#progress-timer");

const resultCard = document.querySelector("#result-card");
const resultTitle = document.querySelector("#result-title");
const resultBadge = document.querySelector("#result-badge");
const resultVerdictTag = document.querySelector("#result-verdict-tag");
const resultRisk = document.querySelector("#result-risk");
const resultExplanation = document.querySelector("#result-explanation");

const contentRiskValue = document.querySelector("#content-risk-value");
const contentRiskFill = document.querySelector("#content-risk-fill");
const mlScoreValue = document.querySelector("#ml-score-value");
const mlScoreFill = document.querySelector("#ml-score-fill");

const findingsList = document.querySelector("#findings-list");
const analysisList = document.querySelector("#analysis-list");
const detailsGrid = document.querySelector("#details-grid");

const historyList = document.querySelector("#history-list");
const historyEmpty = document.querySelector("#history-empty");
const clearHistoryButton = document.querySelector("#clear-history");

const infoEngine = document.querySelector("#info-engine");
const infoModel = document.querySelector("#info-model");
const infoThreat = document.querySelector("#info-threat");
const engineName = document.querySelector("#engine-name");

const serverStatus = document.querySelector("#server-status");
const serverStatusLabel = document.querySelector("#server-status-label");

const navToggle = document.querySelector("#nav-toggle");
const sidebar = document.querySelector("#sidebar");
const sidebarBackdrop = document.querySelector("#sidebar-backdrop");

/* Same page served from the API itself (port 8000) -> relative path.
   Served by Vite (port 5173) or opened as a file -> explicit API origin. */
const isLocalDev =
  location.protocol === "file:" ||
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1";

const API_BASE =
  isLocalDev && location.port !== "8000"
    ? "http://127.0.0.1:8000"
    : "";

const READY_MESSAGE = "Connected to the FraudShield layered inspection engine";

/* ============================================================================
   Server status (real GET /api/health)
============================================================================ */

let healthTimer = null;

function setServerStatus(state, label) {
  serverStatus.dataset.state = state;
  serverStatusLabel.textContent = label;
}

async function checkHealth() {
  setServerStatus("checking", "Checking API");
  try {
    const response = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    setServerStatus("online", "API Online");
    if (data && typeof data.engine === "string" && infoEngine) {
      infoEngine.textContent = data.engine;
    }
    if (data && typeof data.model === "string" && infoModel.textContent === "Loading…") {
      infoModel.textContent = `${data.model} (from /api/health)`;
    }
    if (data && typeof data.model === "string" && engineName) {
      const shortName = data.model.split(" (")[0];
      engineName.textContent = `${shortName} URL model`;
    }
  } catch {
    setServerStatus("offline", "API Offline");
  }
}

function scheduleHealthPolling() {
  checkHealth();
  healthTimer = window.setInterval(checkHealth, 20000);
}

serverStatus.addEventListener("click", () => {
  if (healthTimer) window.clearInterval(healthTimer);
  checkHealth();
  healthTimer = window.setInterval(checkHealth, 20000);
});

window.addEventListener("online", checkHealth);

/* ============================================================================
   Scanning
============================================================================ */

/* Staged descriptions of what the server actually performs. The backend does
   not stream progress, so this cycling text is an honest loading indicator —
   never a fake percentage. */
const PROGRESS_STAGES = [
  "Resolving and validating destination DNS…",
  "Connecting (TLS certificate check)…",
  "Reading page content (read-only)…",
  "Analyzing content, redirects, and model signals…",
];

let stageTimer = null;
let elapsedTimer = null;
let progressStart = 0;
let progressStageIndex = 0;
let isScanning = false;

function setMessage(message, isError = false) {
  scanMessage.textContent = message;
  scanMessage.classList.toggle("is-error", isError);
}

function buildProgressStages() {
  progressStagesList.replaceChildren();
  PROGRESS_STAGES.forEach((text) => {
    const item = document.createElement("li");
    item.className = "progress-stage";
    const dot = document.createElement("span");
    dot.className = "stage-dot";
    dot.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.textContent = text;
    item.append(dot, label);
    item.dataset.stage = text;
    progressStagesList.append(item);
  });
}

function setProgressStage(index) {
  progressStagesList.querySelectorAll(".progress-stage").forEach((stage, i) => {
    stage.classList.toggle("is-done", i < index);
    stage.classList.toggle("is-active", i === index);
  });
}

function startProgress() {
  buildProgressStages();
  progressStageIndex = 0;
  setProgressStage(0);
  scanProgress.hidden = false;
  scanProgress.classList.add("is-live");
  progressStart = performance.now();
  stageTimer = window.setInterval(() => {
    progressStageIndex = Math.min(progressStageIndex + 1, PROGRESS_STAGES.length - 1);
    setProgressStage(progressStageIndex);
  }, 2200);
  elapsedTimer = window.setInterval(updateElapsedDisplay, 100);
  updateElapsedDisplay();
}

function updateElapsedDisplay() {
  const seconds = (performance.now() - progressStart) / 1000;
  progressTimerDisplayElement.textContent = `${seconds.toFixed(1)}s`;
}

function stopProgress() {
  if (stageTimer) window.clearInterval(stageTimer);
  if (elapsedTimer) window.clearInterval(elapsedTimer);
  stageTimer = null;
  elapsedTimer = null;
  scanProgress.hidden = true;
  scanProgress.classList.remove("is-live");
}

function setLoading(isLoading) {
  scanButton.disabled = isLoading;
  scanButton.classList.toggle("is-loading", isLoading);
  scanButton.setAttribute("aria-busy", String(isLoading));
}

function normalizeUrl(value) {
  const trimmedValue = value.trim();
  return /^https?:\/\//i.test(trimmedValue) ? trimmedValue : `https://${trimmedValue}`;
}

/* Full layered scan via the backend inspection engine (unchanged contract). */
async function scanURL(url) {
  const response = await fetch(`${API_BASE}/api/scan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url: url }),
  });

  if (!response.ok) {
    let detailMessage = "Invalid URL";
    try {
      const detail = await response.json();
      const payload = detail && detail.detail;
      if (payload && payload.message) detailMessage = payload.message;
      else if (payload && payload.code) detailMessage = `Rejected (${payload.code})`;
    } catch {
      // keep the default message
    }
    throw new Error(detailMessage);
  }

  const data = await response.json();
  return data;
}

/* ============================================================================
   Verdict presentation
   Backend verdicts: SAFE / SUSPICIOUS / MALICIOUS / INSUFFICIENT EVIDENCE.
   Dashboard badges: SAFE / SUSPICIOUS / HIGH RISK (= MALICIOUS) / UNVERIFIED
   (= INSUFFICIENT EVIDENCE). The exact backend verdict is always shown in the
   small tag next to the badge.
============================================================================ */

const VERDICT_LEVELS = {
  "SAFE": "safe",
  "SUSPICIOUS": "suspicious",
  "MALICIOUS": "malicious",
  "INSUFFICIENT EVIDENCE": "insufficient",
};

const VERDICT_META = {
  safe: {
    icon: "✓", label: "SAFE", risk: "Low",
    explanation: "Full live inspection found no significant risk indicators.",
  },
  suspicious: {
    icon: "⚠", label: "SUSPICIOUS", risk: "Concerning indicators found",
    explanation: "Some inspection signals deserve a closer look before you trust this site.",
  },
  malicious: {
    icon: "⚠", label: "HIGH RISK", risk: "Strong phishing / malware evidence",
    explanation: "Strong evidence of phishing or malicious behavior was observed.",
  },
  insufficient: {
    icon: "◌", label: "UNVERIFIED", risk: "Could not verify — no claim made",
    explanation: "Inspection was incomplete or unavailable (INSUFFICIENT EVIDENCE). No safety claim is made.",
  },
};

const AVAILABILITY_LABELS = {
  "REACHABLE": "reachable",
  "PAGE_NOT_FOUND": "page not found",
  "TIMEOUT": "timed out",
  "CONNECTION_FAILED": "connection failed",
  "DNS_FAILED": "DNS resolution failed",
  "TITLE_UNAVAILABLE": "reachable, no page title",
  "UNKNOWN": "unknown",
};

/* ============================================================================
   Result rendering (only real response data)
============================================================================ */

function setBarFill(fillElement, percent) {
  fillElement.style.setProperty("--fill", `${Math.max(0, Math.min(100, percent))}%`);
}

const severityClass = {
  high: "is-high",
  medium: "is-medium",
  low: "is-low",
};

function severityIcon(severity) {
  if (severity === "high" || severity === "medium") return "⚠";
  if (severity === "low") return "ℹ";
  return "✳";
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function addDetailRow(label, valueHTML) {
  const row = document.createElement("div");
  row.className = "detail-row";
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.innerHTML = valueHTML;
  row.append(dt, dd);
  detailsGrid.append(row);
}

function renderDetails(data, inspection) {
  detailsGrid.replaceChildren();

  const tls = (data.security && data.security.tls) || null;
  const dns = (data.security && data.security.dns) || null;
  const hops = (data.security && data.security.redirects) || [];

  addDetailRow("Scanned URL", `<strong>${escapeHtml(data.url || "—")}</strong>`);
  addDetailRow("Availability", escapeHtml(AVAILABILITY_LABELS[data.availability] || AVAILABILITY_LABELS["UNKNOWN"]));

  if (inspection.final_url && inspection.final_url !== data.url) {
    addDetailRow("Final destination", escapeHtml(inspection.final_url));
  }
  const pageTitle = inspection.page && inspection.page.title;
  addDetailRow(
    "Page title",
    pageTitle ? escapeHtml(pageTitle) : (inspection.reachable ? "No page title was returned." : "—"),
  );

  if (dns && dns.validated_ips && dns.validated_ips.length) {
    addDetailRow(
      "DNS (validated IPs)",
      dns.validated_ips.map(escapeHtml).join("<br>"),
    );
  } else if (dns && dns.error) {
    addDetailRow("DNS", `Resolution failed: ${escapeHtml(dns.error)}`);
  }

  if (tls && tls.handshake === "ok") {
    const tlsBits = [
      `TLS ${escapeHtml(tls.tls_version || "?")}`,
      `cert: ${escapeHtml(tls.subject_cn || "?")}`,
      `issuer: ${escapeHtml(tls.issuer || "?")}`,
    ];
    if (tls.not_after) tlsBits.push(`valid until ${escapeHtml(tls.not_after)}`);
    addDetailRow("TLS certificate", tlsBits.join("<br>"));
  } else if (tls) {
    addDetailRow("TLS certificate", escapeHtml(`Not verified (${tls.handshake ?? "n/a"})`));
  }

  if (hops.length) {
    const hopLines = hops
      .map(
        (hop, i) =>
          `<span class="redirect-line mono">#${i + 1} ${escapeHtml(hop.status ?? "?")} → ${escapeHtml(hop.url || "—")}${
            hop.location ? ` → ${escapeHtml(hop.location)}` : ""
          }</span>`,
      )
      .join("");
    addDetailRow("Redirects", hopLines);
  } else {
    addDetailRow("Redirects", "No redirect chain observed.");
  }

  if (inspection.redirect_scheme_downgrade) {
    addDetailRow("Schema warning", "<strong>HTTPS downgraded to HTTP during redirects.</strong>");
  }
  if (typeof inspection.elapsed_ms === "number") {
    addDetailRow("Inspection duration", `${escapeHtml(String(inspection.elapsed_ms))} ms`);
  }
}

function renderFindings(findings) {
  findingsList.replaceChildren();
  if (!findings.length) {
    const empty = document.createElement("li");
    empty.className = "findings-empty";
    empty.textContent = "No findings were produced by the live inspection.";
    findingsList.append(empty);
    return;
  }
  findings.forEach((finding) => {
    const item = document.createElement("li");
    item.className = `finding-item ${severityClass[finding.severity] || "is-info"}`;
    const sev = document.createElement("span");
    sev.className = "finding-sev";
    sev.textContent = finding.severity || "info";
    const body = document.createElement("span");
    const message = document.createElement("span");
    message.textContent = finding.message || "";
    body.append(message);
    if (finding.code) {
      const code = document.createElement("span");
      code.className = "finding-code";
      code.textContent = finding.code;
      body.append(code);
    }
    item.append(sev, body);
    findingsList.append(item);
  });
}

function renderEvidence(evidence) {
  analysisList.replaceChildren();
  if (!evidence.length) {
    const empty = document.createElement("li");
    empty.className = "evidence-empty";
    empty.textContent = "No evidence lines were returned.";
    analysisList.append(empty);
    return;
  }
  evidence.forEach((line) => {
    if (line.startsWith("Verdict ")) return; // already shown as the badge
    const item = document.createElement("li");
    item.className = "evidence-item";
    const icon = document.createElement("span");
    icon.className = "evidence-icon";
    icon.textContent = severityIcon("info");
    if (line.startsWith("Findings:")) {
      icon.textContent = "⚠";
      item.classList.add("is-warning");
    } else if (line.startsWith("Raw URL model")) {
      item.classList.add("is-neutral");
    }
    item.append(icon, document.createTextNode(line));
    analysisList.append(item);
  });
}

function renderResult(data) {
  const inspection = data.inspection || {};
  const level = VERDICT_LEVELS[data.verdict] || "insufficient";
  const meta = VERDICT_META[level];

  resultCard.dataset.level = level;
  resultTitle.textContent = data.url || "—";
  resultTitle.title = data.url || "";
  resultBadge.textContent = `${meta.icon} ${meta.label}`;
  resultRisk.textContent = meta.risk;
  resultExplanation.textContent = meta.explanation;

  // The exact backend verdict stays visible whenever it differs from the badge label.
  const tagText = `backend verdict: ${data.verdict || "—"}`;
  if (tagText.toUpperCase() !== meta.label.toUpperCase()) {
    resultVerdictTag.hidden = false;
    resultVerdictTag.textContent = tagText;
  } else {
    resultVerdictTag.hidden = true;
    resultVerdictTag.textContent = "";
  }

  /* Content risk: engine composite from real findings (capped 0-100). */
  const signals = data.signals || {};
  const contentRisk = typeof signals.content_risk === "number" ? signals.content_risk : null;
  if (contentRisk !== null) {
    contentRiskValue.textContent = `${contentRisk}/100`;
    setBarFill(contentRiskFill, contentRisk);
  } else {
    contentRiskValue.textContent = "—";
    setBarFill(contentRiskFill, 0);
  }

  /* Raw ML score (R2): url_model.model_score, labelled, never decisive. */
  const model = data.url_model || {};
  if (model.model_score != null) {
    mlScoreValue.textContent = `${model.model_score}`;
    setBarFill(mlScoreFill, model.model_score * 100);
  } else {
    mlScoreValue.textContent = "—";
    setBarFill(mlScoreFill, 0);
  }

  renderFindings(data.findings || []);
  renderEvidence(data.evidence || []);
  renderDetails(data, inspection);

  /* Machine-readable extras from the response -> About panel (real data). */
  if (model.model) {
    infoModel.textContent = `${model.model.name || "R2"} · variant ${model.model.variant || "?"} · ${model.model.feature_count || 14} features`;
  }
  const threat = data.threat_intelligence || {};
  if (infoThreat && typeof threat.available === "boolean") {
    infoThreat.textContent = threat.available ? "Configured" : "Not configured (honest reporting — no feed connected)";
  }

  scanProgress.hidden = true;
  resultCard.hidden = false;
  resultCard.classList.add("is-entering");
  window.setTimeout(() => resultCard.classList.remove("is-entering"), 500);
}

/* ============================================================================
   Session scan history (only genuinely performed scans; no backend store)
============================================================================ */

const HISTORY_MAX = 8;
let scanHistory = [];

function timeLabel(isoString) {
  try {
    return new Date(isoString).toLocaleTimeString();
  } catch {
    return "";
  }
}

function renderHistory() {
  historyList.replaceChildren();
  historyEmpty.hidden = scanHistory.length > 0;
  clearHistoryButton.hidden = scanHistory.length === 0;

  scanHistory.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "history-item";

    const dot = document.createElement("span");
    dot.className = `history-verdict history-verdict--${entry.level}`;
    dot.setAttribute("aria-hidden", "true");

    const main = document.createElement("div");
    main.className = "history-main";
    const urlLine = document.createElement("p");
    urlLine.className = "history-url mono";
    urlLine.textContent = entry.url;
    urlLine.title = entry.url;
    const metaLine = document.createElement("p");
    metaLine.className = "history-meta";
    const scoreText =
      entry.modelScore != null ? ` · raw ML score ${entry.modelScore}` : "";
    metaLine.textContent = `${entry.badgeLabel} · ${timeLabel(entry.at)}${scoreText}`;
    main.append(urlLine, metaLine);

    const badge = document.createElement("span");
    badge.className = `history-badge history-badge--${entry.level}`;
    badge.textContent = entry.badgeLabel;

    const rescan = document.createElement("button");
    rescan.type = "button";
    rescan.className = "history-rescan";
    rescan.textContent = "Re-scan";
    rescan.addEventListener("click", () => {
      urlInput.value = entry.url;
      void handleScan();
    });

    item.append(dot, badge, main, rescan);
    historyList.append(item);
  });
}

function recordHistory(entry) {
  scanHistory.unshift(entry);
  scanHistory = scanHistory.slice(0, HISTORY_MAX);
  renderHistory();
}

clearHistoryButton.addEventListener("click", () => {
  scanHistory = [];
  renderHistory();
});

/* ============================================================================
   Handle a scan
============================================================================ */

async function handleScan() {
  if (isScanning) return; // duplicate submissions not allowed
  const rawUrl = urlInput.value.trim();

  if (!rawUrl) {
    setMessage("Enter a website URL to start your scan.", true);
    urlInput.focus();
    return;
  }

  let normalizedUrl;
  try {
    normalizedUrl = normalizeUrl(rawUrl);
    const parsedUrl = new URL(normalizedUrl);
    if (!["http:", "https:"].includes(parsedUrl.protocol) || !parsedUrl.hostname) {
      throw new TypeError("Please enter a valid website address.");
    }
  } catch {
    setMessage("That URL doesn't look right. Try something like https://example.com.", true);
    urlInput.focus();
    return;
  }

  isScanning = true;
  resultCard.hidden = true;
  setLoading(true);
  startProgress();

  try {
    const data = await scanURL(normalizedUrl);
    stopProgress();
    renderResult(data);
    const level = VERDICT_LEVELS[data.verdict] || "insufficient";
    recordHistory({
      url: data.url || normalizedUrl,
      level,
      badgeLabel: VERDICT_META[level].label,
      modelScore: data.url_model ? data.url_model.model_score : null,
      at: new Date().toISOString(),
    });
    setMessage(
      "Layered scan complete: live website inspection + URL-string model (R2).",
    );
  } catch (error) {
    stopProgress();
    console.error("The website scan could not be completed.", error);
    setMessage(
      error instanceof Error && error.message
        ? `Scan failed: ${error.message}`
        : "The scan could not be completed. Please check the URL and try again.",
      true,
    );
  } finally {
    setLoading(false);
    isScanning = false;
  }
}

scanForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void handleScan();
});

urlInput.addEventListener("input", () => {
  if (scanMessage.classList.contains("is-error")) {
    setMessage(READY_MESSAGE);
  }
});

/* ============================================================================
   Sidebar drawer (mobile)
============================================================================ */

function setSidebarOpen(open) {
  sidebar.classList.toggle("is-open", open);
  sidebarBackdrop.classList.toggle("is-visible", open);
  sidebarBackdrop.hidden = !open;
  navToggle.setAttribute("aria-expanded", String(open));
  navToggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
}

navToggle.addEventListener("click", () => {
  setSidebarOpen(!sidebar.classList.contains("is-open"));
});
sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));

sidebar.querySelectorAll(".side-link").forEach((link) => {
  link.addEventListener("click", () => setSidebarOpen(false));
});

/* ============================================================================
   Init
============================================================================ */

renderHistory();
scheduleHealthPolling();
