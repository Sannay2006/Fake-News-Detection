const articleInput = document.querySelector("#article");
const charCount = document.querySelector("#char-count");
const sampleButton = document.querySelector("#sample-button");
const analyzeButton = document.querySelector("#analyze-button");
const resetButton = document.querySelector("#reset-button");
const formError = document.querySelector("#form-error");
const emptyState = document.querySelector("#empty-state");
const resultState = document.querySelector("#result-state");

const sampleArticle = `The city council approved a new public transit plan on Tuesday after months of hearings and consultation with residents. The proposal adds two bus routes, increases weekend service, and sets aside funding for accessibility upgrades. Officials said the changes will take effect in September. The council will publish quarterly reports on ridership and costs, and plans to review the program after its first year.`;

function updateInputState() {
  const length = articleInput.value.trim().length;
  charCount.textContent = `${articleInput.value.length.toLocaleString()} / 20,000`;
  analyzeButton.disabled = length < 60;
  formError.hidden = true;
}

articleInput.addEventListener("input", updateInputState);

sampleButton.addEventListener("click", () => {
  articleInput.value = sampleArticle;
  updateInputState();
  articleInput.focus();
});

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function renderResult(result) {
  const isFake = result.label === "fake";
  resultState.classList.toggle("is-fake", isFake);
  document.querySelector("#result-label").textContent = isFake ? "Likely fake" : "Likely real";
  document.querySelector("#result-icon").textContent = isFake ? "↙" : "↗";
  document.querySelector("#result-description").textContent = isFake
    ? "The model found language patterns more common in its fake-news examples."
    : "The model found language patterns more common in its real-news examples.";
  document.querySelector("#confidence-value").textContent = `${Math.round(result.confidence * 100)}%`;
  document.querySelector("#confidence-bar").style.width = `${Math.round(result.confidence * 100)}%`;
  document.querySelector("#result-caveat").textContent = result.caveat;

  const signalList = document.querySelector("#signal-list");
  signalList.replaceChildren();
  if (!result.signals.length) {
    const empty = document.createElement("span");
    empty.className = "no-signals";
    empty.textContent = "No strong word signals found in this text.";
    signalList.append(empty);
  } else {
    for (const signal of result.signals) {
      const chip = document.createElement("span");
      chip.className = "signal-chip";
      chip.textContent = signal.term;
      signalList.append(chip);
    }
  }
  emptyState.hidden = true;
  resultState.hidden = false;
}

analyzeButton.addEventListener("click", async () => {
  analyzeButton.disabled = true;
  analyzeButton.querySelector("span:first-child").textContent = "Reading patterns…";
  formError.hidden = true;
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article: articleInput.value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Something went wrong. Please try again.");
    renderResult(result);
  } catch (error) {
    showError(error.message || "Could not reach the analysis service. Is the app running?");
  } finally {
    analyzeButton.querySelector("span:first-child").textContent = "Analyze article";
    analyzeButton.disabled = articleInput.value.trim().length < 60;
  }
});

resetButton.addEventListener("click", () => {
  emptyState.hidden = false;
  resultState.hidden = true;
  articleInput.focus();
});

updateInputState();
