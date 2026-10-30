"use strict";

const scanForm = document.querySelector("#scan-form");
const urlInput = document.querySelector("#url-input");
const scanButton = document.querySelector("#scan-button");
const scanMessage = document.querySelector("#scan-message");
const resultCard = document.querySelector("#result-card");
const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector("#primary-navigation");

const suspiciousTerms = [
  "login",
  "verify",
  "secure",
  "account",
  "update",
  "wallet",
  "banking",
  "password",
  "signin",
  "confirm",
];

function normalizeUrl(value) {
  const trimmedValue = value.trim();
  return /^https?:\/\//i.test(trimmedValue) ? trimmedValue : `https://${trimmedValue}`;
}

function getUrlSignals(parsedUrl) {
  const hostname = parsedUrl.hostname.toLowerCase();
  const fullUrl = parsedUrl.href.toLowerCase();
  const isIpAddress = /^\d{1,3}(?:\.\d{1,3}){3}$/.test(hostname);
  const subdomainCount = Math.max(hostname.split(".").length - 2, 0);
  const suspiciousTerm = suspiciousTerms.find((term) => fullUrl.includes(term));
  const isHttp = parsedUrl.protocol === "http:";
  const hasPunycode = hostname.includes("xn--");
  const isLong = parsedUrl.href.length > 100;
  const hasManySubdomains = subdomainCount > 2;

  let score = 8;
  if (isHttp) score += 18;
  if (suspiciousTerm) score += 26;
  if (isIpAddress) score += 32;
  if (hasManySubdomains) score += 17;
  if (hasPunycode) score += 22;
  if (isLong) score += 10;

  const level = score >= 70 ? "high" : score >= 35 ? "suspicious" : "safe";
  const reasons = [
    {
      warning: isHttp,
      text: isHttp ? "HTTP connection — no HTTPS detected" : "HTTPS connection detected",
    },
    {
      warning: hasManySubdomains || isLong,
      text: hasManySubdomains
        ? "Excessive subdomains"
        : isLong
          ? "Unusually long URL"
          : "Normal URL structure",
    },
    {
      warning: Boolean(suspiciousTerm),
      text: suspiciousTerm
        ? `Suspicious keyword detected: “${suspiciousTerm}”`
        : "No suspicious keywords detected",
    },
    {
      warning: isIpAddress || hasPunycode,
      text: isIpAddress
        ? "IP address used instead of a domain"
        : hasPunycode
          ? "Encoded / unusual domain detected"
          : "No IP address used",
    },
  ];

  return {
    level,
    score: Math.min(score, 99),
    reasons,
  };
}

// Keep this mock function as the single seam to replace with a FastAPI request later.
function scanURL(url) {
  return new Promise((resolve) => {
    window.setTimeout(() => {
      const parsedUrl = new URL(url);
      resolve({
        url: parsedUrl.href,
        ...getUrlSignals(parsedUrl),
      });
    }, 1350);
  });
}

function setLoading(isLoading) {
  scanButton.disabled = isLoading;
  scanButton.classList.toggle("is-loading", isLoading);
  scanButton.setAttribute("aria-busy", String(isLoading));
}

function setMessage(message, isError = false) {
  scanMessage.textContent = message;
  scanMessage.classList.toggle("is-error", isError);
}

function renderResult(result) {
  const resultBadge = document.querySelector("#result-badge");
  const resultRisk = document.querySelector("#result-risk");
  const resultUrl = document.querySelector("#result-url");
  const resultExplanation = document.querySelector("#result-explanation");
  const riskMeter = document.querySelector("#risk-meter");
  const riskScore = document.querySelector("#risk-score");
  const analysisList = document.querySelector("#analysis-list");
  const labels = {
    safe: { icon: "✓", label: "SAFE", risk: "Low Risk", explanation: "No major warning signs stood out in this demo scan." },
    suspicious: { icon: "⚠", label: "SUSPICIOUS", risk: "Potential Risk Detected", explanation: "A few URL patterns deserve a closer look before you continue." },
    high: { icon: "⚠", label: "HIGH RISK", risk: "High Risk Detected", explanation: "Several warning signs were found in this demo scan." },
  };
  const assessment = labels[result.level];

  resultCard.dataset.level = result.level;
  resultBadge.textContent = `${assessment.icon} ${assessment.label}`;
  resultRisk.textContent = assessment.risk;
  resultUrl.textContent = result.url;
  resultUrl.title = result.url;
  resultExplanation.textContent = assessment.explanation;
  riskMeter.style.setProperty("--risk", 0);
  riskMeter.setAttribute("aria-label", `Risk score ${result.score}%`);
  riskScore.textContent = `${result.score}%`;
  analysisList.replaceChildren();

  result.reasons.forEach((reason) => {
    const item = document.createElement("li");
    const icon = document.createElement("span");
    icon.textContent = reason.warning ? "⚠" : "✓";
    item.classList.toggle("is-warning", reason.warning);
    item.append(icon, document.createTextNode(reason.text));
    analysisList.append(item);
  });

  resultCard.hidden = false;
  window.requestAnimationFrame(() => {
    riskMeter.style.setProperty("--risk", result.score);
  });
}

async function handleScan(event) {
  event.preventDefault();
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
    setMessage("That URL doesn’t look right. Try something like https://example.com.", true);
    urlInput.focus();
    return;
  }

  resultCard.hidden = true;
  setLoading(true);
  setMessage("Analyzing URL characteristics…");

  try {
    const result = await scanURL(normalizedUrl);
    renderResult(result);
    setMessage("Demo scan complete · No URL was sent to a server");
  } catch (error) {
    console.error("The demo URL scan could not be completed.", error);
    setMessage("The scan could not be completed. Please check the URL and try again.", true);
  } finally {
    setLoading(false);
  }
}

scanForm.addEventListener("submit", handleScan);

urlInput.addEventListener("input", () => {
  if (scanMessage.classList.contains("is-error")) {
    setMessage("Demo mode · No URL is sent to a server");
  }
});

navToggle.addEventListener("click", () => {
  const isExpanded = navToggle.getAttribute("aria-expanded") === "true";
  navToggle.setAttribute("aria-expanded", String(!isExpanded));
  navToggle.setAttribute("aria-label", isExpanded ? "Open navigation menu" : "Close navigation menu");
  navLinks.classList.toggle("is-open", !isExpanded);
});

navLinks.addEventListener("click", (event) => {
  if (event.target instanceof HTMLAnchorElement) {
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.setAttribute("aria-label", "Open navigation menu");
    navLinks.classList.remove("is-open");
  }
});

const siteHeader = document.querySelector(".site-header");
window.addEventListener(
  "scroll",
  () => siteHeader.classList.toggle("is-scrolled", window.scrollY > 16),
  { passive: true },
);

const revealElements = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 },
  );
  revealElements.forEach((element) => revealObserver.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("is-visible"));
}
