"use strict";

const STORAGE_KEY = "legal-assistant-chat-v1";
const MAX_CLIENT_HISTORY = 24;

const STATUS_INFO = {
  answered: { label: "Подтверждено документами", cls: "ok" },
  insufficient: { label: "Неполный ответ", cls: "warn" },
  refused: { label: "Нет подтверждения", cls: "bad" },
  clarify: { label: "Нужно уточнение", cls: "info" },
  offtopic: { label: "Не по теме", cls: "muted" },
};

const FALLBACK_EXAMPLES = [
  "Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?",
  "Какой максимальный срок испытания можно установить главному бухгалтеру?",
  "Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?",
  "По какому адресу доставляется товар по договору поставки № П-17/2024 после 1 октября 2024 года?",
];

const state = {
  messages: [],
  history: [],
  documents: [],
  examples: [],
  selectedSources: [],
  activeTab: "sources",
  ready: false,
  busy: false,
  health: null,
  currentDoc: null,
  currentChunkId: null,
  lastFocus: null,
  seq: 0,
};

let refs = {};
let toastTimer = null;
let healthTimer = null;

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function newId() {
  state.seq += 1;
  return `m${Date.now()}-${state.seq}`;
}

function formatTime(value) {
  if (!value) return "";
  try {
    return new Date(value).toLocaleTimeString("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
    });
    refs.documentCount.textContent = String(documents.length);
  } catch {
    return "";
  }
}

function formatIsoDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value || ""));
  return match ? `${match[3]}.${match[2]}.${match[1]}` : String(value || "");
}

function statusInfo(status) {
  return STATUS_INFO[status] || STATUS_INFO.insufficient;
}

function statusChip(status) {
  const info = statusInfo(status);
  return make("span", `status-chip ${info.cls}`, info.label);
}

function humanizeDocumentTitle(value) {
  const text = String(value || "").trim();
  if (!text || text !== text.toLocaleUpperCase("ru-RU")) return text;
  const marker = text.indexOf("№");
  const head = (marker >= 0 ? text.slice(0, marker) : text)
    .toLocaleLowerCase("ru-RU")
    .trim();
  if (!head) return text;
  const readable = head.charAt(0).toLocaleUpperCase("ru-RU") + head.slice(1);
  return marker >= 0 ? `${readable} ${text.slice(marker).trim()}` : readable;
}

function plural(count, forms) {
  const value = Math.abs(Number(count)) % 100;
  const last = value % 10;
  if (value > 10 && value < 20) return forms[2];
  if (last === 1) return forms[0];
  if (last >= 2 && last <= 4) return forms[1];
  return forms[2];
}

function showToast(message) {
  refs.toast.textContent = message;
  refs.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    refs.toast.hidden = true;
  }, 3200);
}

function setBanner(message, isError) {
  if (!message) {
    refs.banner.hidden = true;
    refs.banner.textContent = "";
    return;
  }
  refs.banner.hidden = false;
  refs.banner.textContent = message;
  refs.banner.classList.toggle("is-error", Boolean(isError));
}

async function fetchJSON(url, options = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 210000);
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  try {
    const response = await fetch(url, { ...options, headers, signal: controller.signal });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }
    if (!response.ok) {
      const detail = data && (data.detail || data.error);
      const error = new Error(
        typeof detail === "string" ? detail : `Ошибка сервера (HTTP ${response.status})`
      );
      error.status = response.status;
      error.data = data;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      const timeout = new Error("Сервер не ответил за 3,5 минуты");
      timeout.status = 504;
      throw timeout;
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

function updateComposer() {
  const serverReady = state.ready;
  refs.message.disabled = !serverReady;
  refs.send.disabled = !serverReady || state.busy;
  refs.send.setAttribute("aria-busy", state.busy ? "true" : "false");
  refs.composerState.classList.remove("is-error");
  if (!serverReady) {
    refs.composerState.textContent = "Индекс готовится…";
  } else if (state.busy) {
    refs.composerState.textContent = "Ассистент отвечает…";
  } else {
    refs.composerState.textContent = "";
  }
}

function updateMessageCount() {
  const length = refs.message.value.length;
  refs.messageCount.textContent = `${length} / 4000`;
  refs.messageCount.classList.toggle("is-limit", length >= 3800);
}

function setReady(ready) {
  state.ready = Boolean(ready);
  updateComposer();
}

function applyHealth(health) {
  state.health = health;
  const chip = refs.healthChip;
  const text = refs.healthText;
  chip.classList.remove("loading", "ok", "bad");
  if (!health || health.status === "loading") {
    chip.classList.add("loading");
    text.textContent = "Подготовка базы…";
    setBanner("");
    setReady(false);
    return;
  }
  if (!health.ready) {
    chip.classList.add("bad");
    text.textContent = "Ошибка инициализации";
    setBanner(
      health.error || "Не удалось подготовить индекс. Проверьте .env и перезапустите приложение.",
      true
    );
    setReady(false);
    return;
  }
  chip.classList.add("ok");
  text.textContent = "Система готова";
  chip.title = [health.llm_model, health.emb_model].filter(Boolean).join(" · ");
  if (health.warning) {
    setBanner(health.warning, false);
  } else {
    setBanner("");
  }
  setReady(true);
}

async function pollHealth() {
  healthTimer = null;
  let health;
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    health = await response.json();
  } catch {
    health = { status: "error", ready: false, error: "Сервер недоступен" };
  }
  applyHealth(health);
  if (!health.ready) {
    if (!state.messages.length) renderChat();
    scheduleHealth(health.status === "error" ? 3500 : 1600);
    return;
  }
  if (!state.documents.length) {
    await loadDocuments();
  }
  if (!state.messages.length) renderChat();
}

function scheduleHealth(delay) {
  window.clearTimeout(healthTimer);
  healthTimer = window.setTimeout(pollHealth, delay);
}

async function loadMeta() {
  try {
    const data = await fetchJSON("/api/meta");
    if (Array.isArray(data.examples) && data.examples.length) {
      state.examples = data.examples;
      if (!state.messages.length) renderChat();
    }
  } catch {
  }
}

async function loadDocuments() {
  try {
    const data = await fetchJSON("/api/documents");
    state.documents = Array.isArray(data) ? data : [];
    renderDocuments();
  } catch (error) {
    refs.documentsList.replaceChildren(
      make("p", "side-empty", `Не удалось загрузить документы: ${error.message}`)
    );
  }
}

function bindEvents() {
  refs.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = refs.message.value.trim();
    if (!question) return;
    refs.message.value = "";
    autosizeMessage();
    updateMessageCount();
    submitQuestion(question);
  });

  refs.message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      refs.composer.requestSubmit();
    }
  });
  refs.message.addEventListener("input", autosizeMessage);
  refs.message.addEventListener("input", updateMessageCount);

  refs.newChat.addEventListener("click", clearChat);
  refs.tabSources.addEventListener("click", () => setTab("sources"));
  refs.tabDocuments.addEventListener("click", () => setTab("documents"));
  refs.docSearch.addEventListener("input", renderDocuments);
  refs.drawerClose.addEventListener("click", closeDrawer);
  refs.scrollLatest.addEventListener("click", () => scrollToBottom(true));
  refs.chatScroll.addEventListener("scroll", updateScrollButton, { passive: true });

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-close]")) closeDrawer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !refs.drawer.hidden) closeDrawer();
    if (event.key === "Tab" && !refs.drawer.hidden) trapFocus(event);
    if (
      event.key === "/" &&
      document.activeElement !== refs.message &&
      !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)
    ) {
      event.preventDefault();
      refs.message.focus();
    }
  });
}

function autosizeMessage() {
  refs.message.style.height = "auto";
  const height = Math.min(refs.message.scrollHeight, 180);
  refs.message.style.height = `${height}px`;
  refs.message.style.overflowY = refs.message.scrollHeight > 180 ? "auto" : "hidden";
}

function setTab(name) {
  state.activeTab = name === "documents" ? "documents" : "sources";
  const sources = state.activeTab === "sources";
  refs.tabSources.classList.toggle("is-active", sources);
  refs.tabDocuments.classList.toggle("is-active", !sources);
  refs.tabSources.setAttribute("aria-selected", sources ? "true" : "false");
  refs.tabDocuments.setAttribute("aria-selected", sources ? "false" : "true");
  refs.sourcesPanel.hidden = !sources;
  refs.documentsPanel.hidden = sources;
  if (!sources && state.ready && !state.documents.length) {
    loadDocuments();
  }
}

function clearChat() {
  if (state.busy) {
    showToast("Дождитесь ответа на текущий вопрос");
    return;
  }
  if (state.messages.length && !window.confirm("Очистить историю диалога?")) return;
  state.messages = [];
  state.history = [];
  state.selectedSources = [];
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
  }
  renderChat();
  renderSources([]);
  setTab("sources");
  refs.message.focus();
}

function persist() {
  try {
    const messages = state.messages
      .filter((message) => !message.pending)
      .map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        answer: message.answer,
        status: message.status,
        sources: message.sources,
        missing_info: message.missing_info,
        timings: message.timings,
        pipeline: message.pipeline,
        llm_calls: message.llm_calls,
        error: message.error,
        retryText: message.retryText,
        created: message.created,
      }));
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ messages, history: state.history.slice(-40) })
    );
  } catch {
  }
}

function restore() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (Array.isArray(data.messages)) {
      state.messages = data.messages.filter(
        (message) => message && (message.role === "user" || message.role === "assistant")
      );
    }
    if (Array.isArray(data.history)) {
      state.history = data.history.filter(
        (message) =>
          message &&
          (message.role === "user" || message.role === "assistant") &&
          typeof message.content === "string"
      );
    }
    const lastWithSources = [...state.messages]
      .reverse()
      .find((message) => message.role === "assistant" && Array.isArray(message.sources));
    if (lastWithSources) state.selectedSources = lastWithSources.sources;
  } catch {
    state.messages = [];
    state.history = [];
  }
}

async function submitQuestion(rawQuestion) {
  const question = String(rawQuestion || "").trim().slice(0, 4000);
  if (!question) return;
  if (!state.ready) {
    showToast("Индекс ещё готовится — подождите несколько секунд");
    return;
  }
  if (state.busy) return;

  const priorHistory = state.history.slice(-MAX_CLIENT_HISTORY);
  const userMessage = {
    id: newId(),
    role: "user",
    content: question,
    created: Date.now(),
  };
  const pending = {
    id: newId(),
    role: "assistant",
    pending: true,
    created: Date.now(),
  };
  state.messages.push(userMessage, pending);
  state.history.push({ role: "user", content: question });
  state.busy = true;
  state.selectedSources = [];
  renderChat();
  renderSources([]);
  setTab("sources");
  updateComposer();
  scrollToBottom();

  try {
    const data = await fetchJSON("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: question, history: priorHistory }),
    });
    pending.pending = false;
    pending.answer = String(data.answer || "");
    pending.status = data.status || "insufficient";
    pending.sources = Array.isArray(data.sources) ? data.sources : [];
    pending.missing_info = data.missing_info || null;
    pending.timings = data.timings || {};
    pending.pipeline = data.pipeline || "agent";
    pending.llm_calls = data.llm_calls || 0;
    pending.request_id = data.request_id || "";
    state.history.push({ role: "assistant", content: pending.answer });
    state.selectedSources = pending.sources;
    renderChat();
    renderSources(pending.sources);
    if (pending.sources.length) setTab("sources");
    scrollToBottom();
    persist();
  } catch (error) {
    pending.pending = false;
    pending.error = error.message || "Неизвестная ошибка";
    pending.status = "refused";
    pending.retryText = question;
    if (
      state.history.length &&
      state.history[state.history.length - 1].role === "user" &&
      state.history[state.history.length - 1].content === question
    ) {
      state.history.pop();
    }
    renderChat();
    scrollToBottom();
    persist();
    showToast(`Не удалось получить ответ: ${pending.error}`);
    if (error.status === 503) {
      scheduleHealth(1200);
    }
  } finally {
    state.busy = false;
    updateComposer();
    refs.message.focus();
  }
}

function scrollToBottom(smooth = false) {
  window.requestAnimationFrame(() => {
    refs.chatScroll.scrollTo({
      top: refs.chatScroll.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
    refs.scrollLatest.hidden = true;
  });
}

function updateScrollButton() {
  const distance = refs.chatScroll.scrollHeight - refs.chatScroll.scrollTop - refs.chatScroll.clientHeight;
  refs.scrollLatest.hidden = !state.messages.length || distance < 96;
}

function renderChat() {
  refs.chat.replaceChildren();
  if (!state.messages.length) {
    renderEmpty();
    return;
  }
  for (const message of state.messages) {
    refs.chat.append(renderMessage(message));
  }
  updateScrollButton();
}

function renderEmpty() {
  const wrapper = make("div", "empty-state");
  wrapper.append(make("div", "empty-mark", "§"));
  wrapper.append(make("p", "empty-kicker", "Проверяемые ответы"));
  wrapper.append(make("h2", null, "Задайте вопрос по документам"));
  wrapper.append(
    make(
      "p",
      null,
      "Ответ будет опираться только на базу знаний: договоры, регламенты и законы. " +
        "У каждого ответа — ссылка на документ и подтверждающий фрагмент."
    )
  );

  if (!state.ready) {
    const loading = make(
      "div",
      "init-card",
      state.health && state.health.status === "error"
        ? "Не удалось подготовить базу. Проверьте настройки и перезапустите приложение."
        : "База документов готовится к первому запуску. Это может занять до минуты."
    );
    wrapper.append(loading);
  }

  const examples = (state.examples.length ? state.examples : FALLBACK_EXAMPLES)
    .slice(0, 4)
    .map((item) => (typeof item === "string" ? { id: item, question: item } : item));
  const grid = make("div", "examples");
  for (const example of examples) {
    const button = make("button", "example", example.question);
    button.type = "button";
    button.disabled = !state.ready;
    button.addEventListener("click", () => submitQuestion(example.question));
    grid.append(button);
  }
  wrapper.append(grid);
  refs.chat.append(wrapper);
}

function renderMessage(message) {
  const article = make("article", `msg ${message.role}`);
  article.append(make("div", "avatar", message.role === "user" ? "Вы" : "Ю"));
  const main = make("div", "msg-main");
  const head = make("div", "msg-head");
  head.append(make("strong", null, message.role === "user" ? "Вы" : "Ассистент"));
  if (message.created) head.append(make("span", "meta", formatTime(message.created)));
  main.append(head);

  const bubble = make("div", "bubble");
  if (message.role === "user") {
    bubble.append(make("div", "user-text", message.content || ""));
  } else if (message.pending) {
    bubble.append(typingIndicator());
  } else if (message.error) {
    bubble.append(make("p", "answer-error", `Не удалось получить ответ: ${message.error}`));
    const actions = make("div", "msg-actions");
    const retry = make("button", "link-btn", "Повторить запрос");
    retry.type = "button";
    retry.addEventListener("click", () => retryMessage(message));
    actions.append(retry);
    bubble.append(actions);
  } else {
    renderAssistantBubble(bubble, message);
  }
  main.append(bubble);
  article.append(main);
  return article;
}

function renderAssistantBubble(bubble, message) {
  const answer = make("div", "answer");
  const paragraphs = String(message.answer || "").split(/\n{2,}/);
  for (const paragraph of paragraphs) {
    if (paragraph.trim()) answer.append(make("p", null, paragraph.trim()));
  }
  if (!answer.childNodes.length) answer.append(make("p", null, "Ответ пуст"));
  bubble.append(answer);

  if (message.missing_info) {
    bubble.append(make("div", "missing", `Чего не хватает: ${message.missing_info}`));
  }

  const foot = make("div", "msg-foot");
  foot.append(statusChip(message.status));
  const metaParts = [];
  if (Array.isArray(message.sources) && message.sources.length) {
    const count = message.sources.length;
    metaParts.push(`${count} ${plural(count, ["источник", "источника", "источников"])}`);
  }
  if (metaParts.length) foot.append(make("span", "meta", metaParts.join(" · ")));

  const copy = make("button", "link-btn", "Копировать");
  copy.type = "button";
  copy.addEventListener("click", () => copyText(message.answer || ""));
  foot.append(copy);
  bubble.append(foot);

  if (Array.isArray(message.sources) && message.sources.length) {
    const chips = make("div", "source-chips");
    for (const source of message.sources) {
      chips.append(sourceChip(source));
    }
    bubble.append(chips);
  }
}

function typingIndicator() {
  const typing = make("div", "typing");
  typing.setAttribute("aria-label", "Ассистент печатает");
  typing.append(make("span"), make("span"), make("span"));
  return typing;
}

function sourceChip(source) {
  const button = make("button", "source-chip");
  button.type = "button";
  button.append(
    make("span", "chip-doc", humanizeDocumentTitle(source.doc_title || source.doc || "Документ"))
  );
  if (source.locator) button.append(make("span", "locator", source.locator));
  button.addEventListener("click", () => openDocument(source.doc_id, source.chunk_id));
  return button;
}

function retryMessage(message) {
  if (state.busy) return;
  const retryText = message.retryText || "";
  if (!retryText) return;
  const index = state.messages.indexOf(message);
  if (index >= 0) {
    if (
      index > 0 &&
      state.messages[index - 1].role === "user" &&
      state.messages[index - 1].content === retryText
    ) {
      state.messages.splice(index - 1, 2);
    } else {
      state.messages.splice(index, 1);
    }
  }
  submitQuestion(retryText);
}

function renderSources(sources) {
  const items = Array.isArray(sources) ? sources : [];
  state.selectedSources = items;
  refs.sourcesList.replaceChildren();
  refs.sourcesEmpty.hidden = items.length > 0;
  refs.sourcesList.hidden = items.length === 0;
  refs.sourceCount.textContent = String(items.length);
  refs.sourcesSummary.textContent = items.length
    ? `${items.length} ${plural(items.length, ["фрагмент", "фрагмента", "фрагментов"])} из ответа`
    : "Фрагменты из последнего ответа";
  if (!items.length) return;

  for (const source of items) {
    const card = make("button", "source-card");
    card.type = "button";
    card.title = "Открыть документ и подсветить фрагмент";
    const head = make("div", "source-card-head");
    head.append(
      make("span", "source-title", humanizeDocumentTitle(source.doc_title || source.doc || "Документ"))
    );
    card.append(head);
    if (source.locator) card.append(make("div", "source-locator", source.locator));
    if (source.quote) card.append(make("p", "source-quote", source.quote));
    const footer = make("div", "source-card-footer");
    footer.append(make("span", null, "Открыть фрагмент"));
    footer.append(make("span", null, "↗"));
    card.append(footer);
    card.addEventListener("click", () => openDocument(source.doc_id, source.chunk_id));
    refs.sourcesList.append(card);
  }
}

function docBadge(doc) {
  if (doc.doc_type === "law") return { label: "Действующая редакция", cls: "info" };
  const status = String(doc.status || "").toLowerCase();
  if (status.includes("утратил")) return { label: "Утратил силу", cls: "bad" };
  if (status.includes("действ")) return { label: "Действует", cls: "ok" };
  return { label: doc.status || "Документ", cls: "muted" };
}

function renderDocuments() {
  const query = (refs.docSearch.value || "").trim().toLowerCase();
  refs.documentsList.replaceChildren();
  const documents = state.documents.filter((doc) => {
    if (!query) return true;
    const haystack = [doc.title, doc.filename, doc.doc_number, doc.edition, ...(doc.counterparties || [])]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
  if (!documents.length) {
    refs.documentsList.append(
      make("p", "side-empty", query ? "Ничего не найдено" : "Документы ещё не загружены")
    );
    return;
  }
  for (const doc of documents) {
    const card = make("button", "doc-card");
    card.type = "button";
    const title = make("div", "doc-card-head");
    title.append(make("div", "doc-card-title", humanizeDocumentTitle(doc.title || doc.filename)));
    card.append(title);
    const meta = make("div", "doc-card-meta");
    const badge = docBadge(doc);
    meta.append(make("span", `badge ${badge.cls}`, badge.label));
    if (doc.doc_number) meta.append(make("span", null, doc.doc_number));
    if (doc.edition) meta.append(make("span", null, `ред. ${formatIsoDate(doc.edition)}`));
    else if (doc.doc_date) meta.append(make("span", null, formatIsoDate(doc.doc_date)));
    if (Array.isArray(doc.counterparties) && doc.counterparties.length) {
      meta.append(make("span", null, doc.counterparties.join(", ")));
    }
    card.append(meta);
    card.addEventListener("click", () => openDocument(doc.id, null));
    refs.documentsList.append(card);
  }
}

async function openDocument(docId, chunkId) {
  if (!docId) {
    showToast("У источника нет идентификатора документа");
    return;
  }
  state.lastFocus = document.activeElement;
  refs.drawer.hidden = false;
  document.body.classList.add("drawer-open");
  refs.drawerTitle.textContent = "Загрузка документа…";
  refs.drawerMeta.replaceChildren();
  refs.drawerBody.replaceChildren(make("div", "drawer-loading", "Открываем документ…"));
  refs.drawerClose.focus();
  try {
    const doc = await fetchJSON(`/api/documents/${encodeURIComponent(docId)}`);
    state.currentDoc = doc;
    state.currentChunkId = chunkId || null;
    renderDrawer(doc, chunkId);
  } catch (error) {
    refs.drawerTitle.textContent = "Не удалось открыть документ";
    refs.drawerBody.replaceChildren(make("p", "answer-error", error.message));
  }
}

function renderDrawer(doc, chunkId) {
  state.currentDoc = doc;
  state.currentChunkId = chunkId || null;
  refs.drawerTitle.textContent = humanizeDocumentTitle(doc.title || doc.filename || "Документ");
  refs.drawerMeta.replaceChildren();
  const badge = docBadge(doc);
  refs.drawerMeta.append(make("span", `badge ${badge.cls}`, badge.label));
  if (doc.doc_number) refs.drawerMeta.append(make("span", "meta", doc.doc_number));
  if (doc.edition) refs.drawerMeta.append(make("span", "meta", `редакция от ${formatIsoDate(doc.edition)}`));
  else if (doc.doc_date) refs.drawerMeta.append(make("span", "meta", `дата ${formatIsoDate(doc.doc_date)}`));
  if (Array.isArray(doc.counterparties) && doc.counterparties.length) {
    refs.drawerMeta.append(make("span", "meta", doc.counterparties.join(", ")));
  }
  if (doc.filename) refs.drawerMeta.append(make("span", "meta", doc.filename));

  refs.drawerBody.replaceChildren();
  refs.drawerBody.scrollTop = 0;
  const text = String(doc.text || "");
  const chunk = (doc.chunks || []).find((item) => item.chunk_id === chunkId) || null;
  const docText = make("div", "doc-text");

  let start = chunk && Number.isInteger(chunk.start) ? chunk.start : -1;
  let end = chunk && Number.isInteger(chunk.end) ? chunk.end : -1;
  const quote = chunk ? String(chunk.quote || "") : "";
  const hasRange = start >= 0 && end > start;
  if (chunk && !(hasRange && text.slice(start, end) === quote)) {
    const found = quote ? text.indexOf(quote) : -1;
    if (found >= 0) {
      start = found;
      end = found + quote.length;
    } else if (!hasRange) {
      start = -1;
      end = -1;
    }
  }

  if (start >= 0 && end > start) {
    docText.append(document.createTextNode(text.slice(0, start)));
    const mark = make("mark", null, text.slice(start, end));
    mark.id = "doc-highlight";
    docText.append(mark);
    docText.append(document.createTextNode(text.slice(end)));
  } else {
    docText.textContent = text;
  }
  refs.drawerBody.append(docText);

  if (chunk && !(start >= 0 && end > start)) {
    const box = make("div", "chunk-box");
    box.append(make("div", "chunk-box-title", `Фрагмент${chunk.locator ? ` · ${chunk.locator}` : ""}`));
    box.append(make("div", null, quote || "Текст фрагмента недоступен"));
    refs.drawerBody.append(box);
  }

  renderChunkNav(doc, chunkId);

  window.requestAnimationFrame(() => {
    const mark = document.getElementById("doc-highlight");
    if (mark) {
      mark.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  });
}

function renderChunkNav(doc, chunkId) {
  const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];
  if (!chunks.length) return;
  const nav = make("div", "chunk-nav");
  nav.append(make("h3", null, "Фрагменты документа"));
  const buttons = make("div", "chunk-buttons");
  for (const [index, chunk] of chunks.entries()) {
    const sameAsDocument = chunk.locator && chunk.locator === doc.doc_number;
    const label = chunk.locator && !sameAsDocument ? chunk.locator : `фрагмент ${index + 1}`;
    const button = make(
      "button",
      `chunk-btn${chunk.chunk_id === chunkId ? " is-active" : ""}`,
      label
    );
    button.type = "button";
    button.title = String(chunk.quote || "").slice(0, 160);
    button.addEventListener("click", () => renderDrawer(doc, chunk.chunk_id));
    buttons.append(button);
  }
  nav.append(buttons);
  refs.drawerBody.append(nav);
}

function closeDrawer() {
  refs.drawer.hidden = true;
  document.body.classList.remove("drawer-open");
  state.currentDoc = null;
  state.currentChunkId = null;
  if (state.lastFocus && typeof state.lastFocus.focus === "function") {
    state.lastFocus.focus();
  }
}

function trapFocus(event) {
  const focusable = Array.from(
    refs.drawer.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
  ).filter((node) => node.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

async function copyText(text) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("Ответ скопирован");
    return;
  } catch {
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  try {
    document.execCommand("copy");
    showToast("Ответ скопирован");
  } catch {
    showToast("Не удалось скопировать ответ");
  } finally {
    area.remove();
  }
}

function init() {
  refs = {
    chat: document.getElementById("chat"),
    chatScroll: document.getElementById("chat-scroll"),
    composer: document.getElementById("composer"),
    message: document.getElementById("message"),
    send: document.getElementById("send"),
    composerState: document.getElementById("composer-state"),
    healthChip: document.getElementById("health-chip"),
    healthText: document.getElementById("health-text"),
    banner: document.getElementById("banner"),
    newChat: document.getElementById("new-chat"),
    tabSources: document.getElementById("tab-sources"),
    tabDocuments: document.getElementById("tab-documents"),
    sourcesPanel: document.getElementById("sources-panel"),
    documentsPanel: document.getElementById("documents-panel"),
    sourcesEmpty: document.getElementById("sources-empty"),
    sourcesList: document.getElementById("sources-list"),
    sourceCount: document.getElementById("source-count"),
    sourcesSummary: document.getElementById("sources-summary"),
    docSearch: document.getElementById("doc-search"),
    documentsList: document.getElementById("documents-list"),
    documentCount: document.getElementById("document-count"),
    scrollLatest: document.getElementById("scroll-latest"),
    messageCount: document.getElementById("message-count"),
    drawer: document.getElementById("drawer"),
    drawerClose: document.getElementById("drawer-close"),
    drawerTitle: document.getElementById("drawer-title"),
    drawerMeta: document.getElementById("drawer-meta"),
    drawerBody: document.getElementById("drawer-body"),
    toast: document.getElementById("toast"),
  };

  bindEvents();
  restore();
  renderChat();
  renderSources(state.selectedSources);
  setTab("sources");
  updateComposer();
  autosizeMessage();
  updateMessageCount();
  loadMeta();
  pollHealth();
}

document.addEventListener("DOMContentLoaded", init);

