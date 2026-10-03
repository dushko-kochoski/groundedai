"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  key: "",
  config: null,
  offset: 0,
  total: 0,
  busy: false,
  connected: false,
};
const pageSize = 10;
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = error ? "error" : "";
}
function setBusy(busy) {
  state.busy = busy;
  for (const id of ["upload-button", "ask-button", "refresh"])
    $(id).disabled = busy || !state.connected;
  $("previous").disabled = busy || !state.connected || state.offset === 0;
  $("next").disabled =
    busy || !state.connected || state.offset + pageSize >= state.total;
  document.querySelectorAll(".delete").forEach((button) => {
    button.disabled = busy || !state.connected;
  });
  for (const button of $("access-form").querySelectorAll("button"))
    button.disabled = busy;
}
function clearWorkspace() {
  state.connected = false;
  state.config = null;
  state.offset = 0;
  state.total = 0;
  $("documents").replaceChildren();
  $("document-count").textContent = "—";
  $("page-label").textContent = "";
  $("library-state").textContent = "Connect to load your documents.";
  $("answer-result").hidden = true;
  $("empty-answer").hidden = false;
  $("answer-text").textContent = "";
  $("citations").replaceChildren();
  $("diagnostics").replaceChildren();
  $("question").value = "";
  $("document-file").value = "";
  $("connection").textContent = "Disconnected";
  $("provider-label").textContent = "No workspace connected.";
}
async function api(path, options = {}) {
  const headers = new Headers(options.headers);
  if (state.key) headers.set("Authorization", `Bearer ${state.key}`);
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    headers,
    cache: "no-store",
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401) {
      state.key = "";
      clearWorkspace();
      $("access-panel").open = true;
    }
    const message =
      body?.error?.message ||
      (response.status === 422
        ? "Check the supplied values and try again."
        : `Request failed (${response.status}).`);
    throw new Error(message);
  }
  return body;
}
async function run(operation) {
  if (state.busy) return;
  setBusy(true);
  notice("");
  try {
    await operation();
  } catch (error) {
    notice(error.message || "Could not reach the workspace.", true);
  } finally {
    setBusy(false);
  }
}
function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function loadDocuments() {
  let data = await api(`/documents?limit=${pageSize}&offset=${state.offset}`);
  if (state.offset > 0 && !data.items.length) {
    state.offset = Math.max(
      0,
      Math.floor((data.total - 1) / pageSize) * pageSize,
    );
    data = await api(`/documents?limit=${pageSize}&offset=${state.offset}`);
  }
  state.total = data.total;
  $("document-count").textContent = String(data.total);
  $("library-state").textContent = data.total
    ? ""
    : "No documents yet. Upload your first source above.";
  $("page-label").textContent = data.total
    ? `${state.offset + 1}–${state.offset + data.items.length} of ${data.total}`
    : "";
  $("documents").replaceChildren();
  for (const doc of data.items) {
    const row = document.createElement("li");
    const info = element("div", "", "document-info");
    info.append(element("span", doc.filename, "document-name"));
    const evaluation = doc.metadata?.evaluation_fixture
      ? " · Evaluation only"
      : "";
    info.append(
      element(
        "span",
        `${doc.chunk_count} chunks · ${doc.status}${evaluation}`,
        "document-meta",
      ),
    );
    const remove = element("button", "Delete", "text-button delete");
    remove.type = "button";
    remove.setAttribute("aria-label", `Delete ${doc.filename}`);
    remove.addEventListener("click", () => {
      if (
        state.busy ||
        !confirm(`Delete “${doc.filename}” and its stored chunks?`)
      )
        return;
      run(async () => {
        await api(`/documents/${doc.id}`, { method: "DELETE" });
        await loadDocuments();
        notice("Document deleted.");
      });
    });
    row.append(info, remove);
    $("documents").append(row);
  }
}
async function connect() {
  clearWorkspace();
  const config = await api("/workspace");
  state.config = config;
  state.connected = true;
  $("question").maxLength = config.max_query_chars;
  $("upload-hint").textContent =
    `TXT, Markdown, text-PDF or DOCX · up to ${(config.max_upload_bytes / 1048576).toFixed(1)} MiB`;
  $("provider-label").textContent =
    `Embeddings: ${config.embedding_provider} · Generation: ${config.generation_provider}`;
  $("connection").textContent = config.access_protected
    ? "Protected workspace"
    : "Local development";
  $("access-panel").open = false;
  await loadDocuments();
  if (
    config.embedding_provider === "deterministic" ||
    config.generation_provider === "deterministic"
  )
    notice(
      "Development providers are active: deterministic retrieval/generation may miss meaning and paraphrases.",
    );
}
$("access-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) return;
  state.key = $("access-key").value;
  $("access-key").value = "";
  run(connect);
});
$("disconnect").addEventListener("click", () => {
  state.key = "";
  $("access-key").value = "";
  clearWorkspace();
  setBusy(false);
  notice("Disconnected. Your access key has been cleared from this page.");
});
$("refresh").addEventListener("click", () => run(loadDocuments));
$("previous").addEventListener("click", () =>
  run(async () => {
    state.offset -= pageSize;
    await loadDocuments();
  }),
);
$("next").addEventListener("click", () =>
  run(async () => {
    state.offset += pageSize;
    await loadDocuments();
  }),
);
$("upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.connected) return;
  run(async () => {
    const file = $("document-file").files[0];
    if (!file) throw new Error("Choose a document first.");
    if (file.size > state.config.max_upload_bytes)
      throw new Error("This file exceeds the workspace upload limit.");
    const form = new FormData();
    form.append("file", file);
    notice("Processing your document…");
    await api("/documents", { method: "POST", body: form });
    $("document-file").value = "";
    state.offset = 0;
    await loadDocuments();
    notice("Document ready. You can now ask questions about it.");
  });
});
function showAnswer(result) {
  $("empty-answer").hidden = true;
  $("answer-result").hidden = false;
  $("answer-status").textContent = result.abstained
    ? "Insufficient evidence"
    : "Sources cited";
  $("answer-text").textContent = result.abstained
    ? result.abstention_reason ||
      "The sources do not provide enough evidence to answer."
    : result.answer;
  $("citations").replaceChildren();
  $("citations-section").hidden = !result.citations.length;
  for (const citation of result.citations) {
    const location = citation.page_number
      ? ` · Page ${citation.page_number}`
      : citation.section
        ? ` · ${citation.section}`
        : "";
    $("citations").append(
      element(
        "li",
        `${citation.source_id} · ${citation.filename}${location} · Similarity ${citation.score.toFixed(3)}`,
      ),
    );
  }
  const d = result.diagnostics;
  const details = {
    "Run ID": result.run_id,
    Generator: `${d.provider} / ${d.model}`,
    "Retrieved chunks": d.retrieved_count,
    "Total time": `${d.total_ms.toFixed(0)} ms`,
    "Top evidence score": d.top_evidence_score ?? "Not available",
    "Evidence threshold": d.evidence_min_score ?? "Not configured",
    "Abstention category": d.abstention_category ?? "None",
    Tokens: d.total_tokens ?? "Not reported",
  };
  $("diagnostics").replaceChildren();
  for (const [name, value] of Object.entries(details))
    $("diagnostics").append(element("dt", name), element("dd", String(value)));
}
$("query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.connected) return;
  run(async () => {
    const question = $("question").value.trim();
    if (!question) throw new Error("Enter a question first.");
    $("answer-region").setAttribute("aria-busy", "true");
    $("answer-result").hidden = true;
    $("empty-answer").hidden = false;
    $("ask-button").textContent = "Checking evidence…";
    try {
      const result = await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      showAnswer(result);
    } finally {
      $("answer-region").setAttribute("aria-busy", "false");
      $("ask-button").textContent = "Find an answer ↗";
    }
  });
});
run(connect);
