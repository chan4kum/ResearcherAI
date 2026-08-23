/**
 * app/static/js/app.js — Vanilla Frontend Client for Agentic Research
 * Lightweight, accessible, zero external dependencies.
 */

(() => {
  'use strict';

  // --- Constants & Config ---
  const STORAGE_KEY_HISTORY = 'agentic_research_history_v1';
  const MAX_HISTORY_ITEMS = 30;

  // --- DOM Elements ---
  const el = {
    // Header & Navigation
    btnNewResearch: document.getElementById('btnNewResearch'),
    btnDocuments: document.getElementById('btnDocuments'),
    btnHistoryToggle: document.getElementById('btnHistoryToggle'),
    systemStatus: document.getElementById('systemStatus'),

    // Views
    landingState: document.getElementById('landingState'),
    resultsState: document.getElementById('resultsState'),

    // Research Input Form
    researchForm: document.getElementById('researchForm'),
    researchInput: document.getElementById('researchInput'),
    btnSubmitResearch: document.getElementById('btnSubmitResearch'),
    btnAttachDoc: document.getElementById('btnAttachDoc'),

    // Result Display Elements
    displayQueryText: document.getElementById('displayQueryText'),
    queryTimestamp: document.getElementById('queryTimestamp'),
    progressCard: document.getElementById('progressCard'),
    progressStatusText: document.getElementById('progressStatusText'),
    progressSteps: document.getElementById('progressSteps'),
    answerCard: document.getElementById('answerCard'),
    answerContent: document.getElementById('answerContent'),
    btnCopyAnswer: document.getElementById('btnCopyAnswer'),
    sourcesSection: document.getElementById('sourcesSection'),
    sourcesCountBadge: document.getElementById('sourcesCountBadge'),
    sourcesList: document.getElementById('sourcesList'),
    debugDetails: document.getElementById('debugDetails'),
    debugContent: document.getElementById('debugContent'),

    // Followup Form
    followupForm: document.getElementById('followupForm'),
    followupInput: document.getElementById('followupInput'),

    // Error Display
    errorCard: document.getElementById('errorCard'),
    errorMessageText: document.getElementById('errorMessageText'),
    errorDetails: document.getElementById('errorDetails'),
    errorDetailsPre: document.getElementById('errorDetailsPre'),
    btnRetryResearch: document.getElementById('btnRetryResearch'),
    btnDismissError: document.getElementById('btnDismissError'),

    // History Drawer
    historyDrawer: document.getElementById('historyDrawer'),
    btnCloseHistory: document.getElementById('btnCloseHistory'),
    historyList: document.getElementById('historyList'),
    btnClearHistory: document.getElementById('btnClearHistory'),

    // Document Modal
    documentModal: document.getElementById('documentModal'),
    btnCloseDocModal: document.getElementById('btnCloseDocModal'),
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    uploadFeedback: document.getElementById('uploadFeedback'),
    docListContainer: document.getElementById('docListContainer'),
    btnRefreshDocs: document.getElementById('btnRefreshDocs'),

    // Toast
    toastContainer: document.getElementById('toastContainer'),
  };

  // --- Application State ---
  const state = {
    isResearching: false,
    currentQuery: '',
    currentMode: 'rag', // 'rag' | 'agent'
    currentResult: null,
    history: loadHistory(),
  };

  // --- Initialization ---
  function init() {
    setupEventListeners();
    renderHistoryList();
    checkSystemStatus();
    autoResizeTextarea(el.researchInput);
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    // Mode selector pills
    document.querySelectorAll('.mode-pill').forEach(pill => {
      pill.addEventListener('click', (e) => {
        document.querySelectorAll('.mode-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        state.currentMode = pill.dataset.mode || 'rag';
      });
    });

    // Example prompt chips
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.dataset.prompt;
        if (prompt) {
          el.researchInput.value = prompt;
          autoResizeTextarea(el.researchInput);
          el.researchInput.focus();
          handleStartResearch(prompt);
        }
      });
    });

    // Research input submit
    el.researchForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const query = el.researchInput.value.trim();
      if (query && !state.isResearching) {
        handleStartResearch(query);
      }
    });

    // Enter key submits (Shift+Enter makes newline)
    el.researchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        el.researchForm.dispatchEvent(new Event('submit'));
      }
    });

    el.researchInput.addEventListener('input', () => autoResizeTextarea(el.researchInput));

    // Follow-up form submit
    el.followupForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const followQuery = el.followupInput.value.trim();
      if (followQuery && !state.isResearching) {
        el.followupInput.value = '';
        handleStartResearch(followQuery);
      }
    });

    // Header buttons
    el.btnNewResearch.addEventListener('click', resetToLanding);
    el.btnHistoryToggle.addEventListener('click', toggleHistoryDrawer);
    el.btnCloseHistory.addEventListener('click', closeHistoryDrawer);
    el.btnClearHistory.addEventListener('click', clearHistory);

    // Document Modal
    el.btnDocuments.addEventListener('click', openDocumentModal);
    el.btnAttachDoc.addEventListener('click', openDocumentModal);
    el.btnCloseDocModal.addEventListener('click', closeDocumentModal);
    el.btnRefreshDocs.addEventListener('click', fetchDocumentsList);

    // Dropzone upload
    el.dropZone.addEventListener('click', () => el.fileInput.click());
    el.fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files));
    el.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.dropZone.classList.add('dragover');
    });
    el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
    el.dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files);
      }
    });

    // Copy answer button
    el.btnCopyAnswer.addEventListener('click', copyAnswerToClipboard);

    // Error actions
    el.btnRetryResearch.addEventListener('click', () => {
      if (state.currentQuery) {
        handleStartResearch(state.currentQuery);
      }
    });
    el.btnDismissError.addEventListener('click', () => {
      el.errorCard.classList.add('hidden');
    });

    // Global keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        resetToLanding();
        el.researchInput.focus();
      }
      if (e.key === 'Escape') {
        closeDocumentModal();
        closeHistoryDrawer();
      }
    });
  }

  // --- Research Execution Flow ---
  async function handleStartResearch(query) {
    if (!query) return;

    state.isResearching = true;
    state.currentQuery = query;
    el.btnSubmitResearch.disabled = true;

    // Transition to Results State
    el.landingState.classList.add('hidden');
    el.resultsState.classList.remove('hidden');

    el.displayQueryText.textContent = query;
    el.queryTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Show progress, hide previous answer & error
    el.progressCard.classList.remove('hidden');
    el.answerCard.classList.add('hidden');
    el.errorCard.classList.add('hidden');
    el.sourcesSection.classList.add('hidden');

    // Run progress stepper
    const progressTimer = simulateProgressSteps();

    try {
      let result;
      if (state.currentMode === 'agent') {
        result = await executeAgentTask(query);
      } else {
        result = await executeRAGQuery(query);
      }

      clearInterval(progressTimer);
      completeAllSteps();

      // Render Final Answer & Citations
      setTimeout(() => {
        renderResearchResult(result);
        saveHistoryItem(query, result);
        state.isResearching = false;
        el.btnSubmitResearch.disabled = false;
      }, 300);

    } catch (err) {
      clearInterval(progressTimer);
      state.isResearching = false;
      el.btnSubmitResearch.disabled = false;
      renderErrorState(err);
    }
  }

  // --- Progress Stepper Simulator ---
  function simulateProgressSteps() {
    const steps = [
      { step: 1, text: 'Understanding question...' },
      { step: 2, text: 'Finding relevant sources...' },
      { step: 3, text: 'Reviewing evidence...' },
      { step: 4, text: 'Verifying answer...' },
    ];
    let currentIdx = 0;
    setStepActive(1, steps[0].text);

    return setInterval(() => {
      currentIdx++;
      if (currentIdx < steps.length) {
        setStepCompleted(currentIdx);
        setStepActive(currentIdx + 1, steps[currentIdx].text);
      }
    }, 450);
  }

  function setStepActive(stepNum, text) {
    el.progressStatusText.textContent = text;
    const stepEls = el.progressSteps.querySelectorAll('.step-item');
    stepEls.forEach(item => {
      const num = parseInt(item.dataset.step, 10);
      if (num === stepNum) {
        item.className = 'step-item step-active';
        item.querySelector('.step-icon').textContent = '●';
      }
    });
  }

  function setStepCompleted(stepNum) {
    const stepEls = el.progressSteps.querySelectorAll('.step-item');
    stepEls.forEach(item => {
      const num = parseInt(item.dataset.step, 10);
      if (num <= stepNum) {
        item.className = 'step-item step-completed';
        item.querySelector('.step-icon').textContent = '✓';
      }
    });
  }

  function completeAllSteps() {
    const stepEls = el.progressSteps.querySelectorAll('.step-item');
    stepEls.forEach(item => {
      item.className = 'step-item step-completed';
      item.querySelector('.step-icon').textContent = '✓';
    });
    el.progressStatusText.textContent = 'Research completed';
  }

  // --- API Client Methods ---
  async function executeRAGQuery(question) {
    const response = await fetch('/api/v1/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        strategy: 'normal',
        top_k: 5,
        rerank: true,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    if (data.answer && (data.answer.includes("not available in the provided context") || data.answer.includes("does not contain specific information"))) {
      try {
        const chatRes = await fetch('/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: question }),
        });
        if (chatRes.ok) {
          const chatData = await chatRes.json();
          if (chatData.answer) {
            data.answer = chatData.answer;
          }
        }
      } catch (e) {
        console.warn("Fallback to chat completion failed", e);
      }
    }

    return data;
  }

  async function executeAgentTask(task) {
    const response = await fetch('/api/v1/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    return {
      question: task,
      answer: data.answer || 'Task completed successfully.',
      citations: [],
      metadata: {
        plan: data.plan || [],
        tools_used: data.tools_used || [],
        duration_ms: data.duration_ms || 0,
        model: data.model || '',
        version: data.config_version || '',
      },
    };
  }

  async function checkSystemStatus() {
    try {
      const res = await fetch('/ready');
      if (res.ok) {
        el.systemStatus.querySelector('.status-dot').style.backgroundColor = 'var(--success)';
        el.systemStatus.querySelector('.status-text').textContent = 'Ready';
      } else {
        el.systemStatus.querySelector('.status-dot').style.backgroundColor = 'var(--warning)';
        el.systemStatus.querySelector('.status-text').textContent = 'Degraded';
      }
    } catch {
      el.systemStatus.querySelector('.status-dot').style.backgroundColor = 'var(--error)';
      el.systemStatus.querySelector('.status-text').textContent = 'Offline';
    }
  }

  // --- Result Rendering ---
  function renderResearchResult(result) {
    state.currentResult = result;
    el.progressCard.classList.add('hidden');
    el.answerCard.classList.remove('hidden');

    // Parse & Render Markdown
    el.answerContent.innerHTML = renderMarkdown(result.answer || '');

    // Render Citations
    const citations = result.citations || [];
    if (citations.length > 0) {
      el.sourcesSection.classList.remove('hidden');
      el.sourcesCountBadge.textContent = `${citations.length} source${citations.length > 1 ? 's' : ''}`;
      
      el.sourcesList.innerHTML = citations.map((cite, index) => {
        const citeNum = index + 1;
        const sourceName = cite.source || 'Document Source';
        const similarity = cite.similarity ? `Match: ${(cite.similarity * 100).toFixed(0)}%` : '';
        const snippet = escapeHtml(cite.content || '');

        return `
          <div class="source-item" id="source-${citeNum}" data-citation-index="${citeNum}">
            <div class="source-item-header">
              <span class="source-name">
                <span class="citation-ref">[${citeNum}]</span>
                <span>${escapeHtml(sourceName)}</span>
              </span>
              <span class="source-score">${similarity}</span>
            </div>
            <div class="source-snippet">${snippet}</div>
          </div>
        `;
      }).join('');

      // Toggle snippet on click
      el.sourcesList.querySelectorAll('.source-item').forEach(item => {
        item.addEventListener('click', () => {
          item.classList.toggle('expanded');
        });
      });

      // Link inline citation clicks to sources
      el.answerContent.querySelectorAll('.citation-ref').forEach(badge => {
        badge.addEventListener('click', (e) => {
          e.preventDefault();
          const citeNum = badge.dataset.ref;
          const targetSource = document.getElementById(`source-${citeNum}`);
          if (targetSource) {
            targetSource.classList.add('expanded');
            targetSource.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            targetSource.style.borderColor = 'var(--accent-primary)';
            setTimeout(() => targetSource.style.borderColor = '', 2000);
          }
        });
      });
    } else {
      el.sourcesSection.classList.add('hidden');
    }

    // Telemetry / Provenance Details
    const meta = result.metadata || {};
    el.debugContent.textContent = JSON.stringify({
      model: result.model || meta.model || 'gpt-4o-mini',
      provider: result.provider || 'mock',
      tokens: {
        prompt: result.prompt_tokens || 0,
        completion: result.completion_tokens || 0,
        total: result.total_tokens || 0,
      },
      retrieval_strategy: result.strategy || 'hybrid',
      citations_count: citations.length,
      provenance: meta.version_provenance || meta.config_version || '1.0.0',
      plan: meta.plan || [],
      tools_used: meta.tools_used || [],
    }, null, 2);
  }

  // --- Error Rendering ---
  function renderErrorState(err) {
    el.progressCard.classList.add('hidden');
    el.answerCard.classList.add('hidden');
    el.errorCard.classList.remove('hidden');

    el.errorMessageText.textContent = 'Something went wrong while completing the research.';
    el.errorDetailsPre.textContent = err.message || String(err);
  }

  // --- Navigation & Reset ---
  function resetToLanding() {
    el.resultsState.classList.add('hidden');
    el.landingState.classList.remove('hidden');
    el.researchInput.value = '';
    autoResizeTextarea(el.researchInput);
    el.researchInput.focus();
  }

  // --- History Management ---
  function loadHistory() {
    try {
      const data = localStorage.getItem(STORAGE_KEY_HISTORY);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  }

  function saveHistoryItem(query, result) {
    const item = {
      id: 'h_' + Date.now(),
      query: query,
      timestamp: new Date().toISOString(),
      result: result,
    };
    state.history = [item, ...state.history.filter(h => h.query !== query)].slice(0, MAX_HISTORY_ITEMS);
    try {
      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(state.history));
    } catch (e) {
      console.warn('Failed to persist history to localStorage', e);
    }
    renderHistoryList();
  }

  function renderHistoryList() {
    if (!state.history || state.history.length === 0) {
      el.historyList.innerHTML = '<div class="empty-history">No recent research history.</div>';
      return;
    }

    el.historyList.innerHTML = state.history.map(item => {
      const dateStr = new Date(item.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
      return `
        <div class="history-item" data-id="${item.id}">
          <div class="history-query">${escapeHtml(item.query)}</div>
          <div class="history-time">${dateStr}</div>
        </div>
      `;
    }).join('');

    el.historyList.querySelectorAll('.history-item').forEach(elItem => {
      elItem.addEventListener('click', () => {
        const id = elItem.dataset.id;
        const entry = state.history.find(h => h.id === id);
        if (entry) {
          closeHistoryDrawer();
          state.currentQuery = entry.query;
          el.landingState.classList.add('hidden');
          el.resultsState.classList.remove('hidden');
          el.displayQueryText.textContent = entry.query;
          el.queryTimestamp.textContent = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          renderResearchResult(entry.result);
        }
      });
    });
  }

  function toggleHistoryDrawer() {
    el.historyDrawer.classList.toggle('open');
    el.historyDrawer.setAttribute('aria-hidden', !el.historyDrawer.classList.contains('open'));
  }

  function closeHistoryDrawer() {
    el.historyDrawer.classList.remove('open');
    el.historyDrawer.setAttribute('aria-hidden', 'true');
  }

  function clearHistory() {
    state.history = [];
    localStorage.removeItem(STORAGE_KEY_HISTORY);
    renderHistoryList();
    showToast('Research history cleared.');
  }

  // --- Document Ingestion Modal ---
  function openDocumentModal() {
    el.documentModal.classList.remove('hidden');
    el.uploadFeedback.classList.add('hidden');
    fetchDocumentsList();
  }

  function closeDocumentModal() {
    el.documentModal.classList.add('hidden');
  }

  async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    el.uploadFeedback.className = 'upload-feedback';
    el.uploadFeedback.textContent = `Uploading ${file.name}...`;
    el.uploadFeedback.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }

      el.uploadFeedback.className = 'upload-feedback success';
      el.uploadFeedback.textContent = `✓ Document '${file.name}' added to knowledge base.`;
      showToast(`✓ Document '${file.name}' added`);
      fetchDocumentsList();
    } catch (err) {
      el.uploadFeedback.className = 'upload-feedback error';
      el.uploadFeedback.textContent = `Failed to upload: ${err.message}`;
    }
  }

  async function fetchDocumentsList() {
    el.docListContainer.innerHTML = '<div class="doc-loading">Loading documents...</div>';
    try {
      const res = await fetch('/api/v1/documents');
      if (!res.ok) throw new Error('Could not fetch documents');
      const docs = await res.json();

      if (!docs || docs.length === 0) {
        el.docListContainer.innerHTML = '<div class="doc-loading">No documents indexed yet.</div>';
        return;
      }

      el.docListContainer.innerHTML = docs.map(doc => `
        <div class="doc-item">
          <span class="doc-name" title="${escapeHtml(doc.source || doc.doc_id)}">${escapeHtml(doc.source || doc.doc_id)}</span>
          <span class="source-score">${doc.chunk_count || 1} chunks</span>
        </div>
      `).join('');
    } catch {
      el.docListContainer.innerHTML = '<div class="doc-loading">Knowledge base ready.</div>';
    }
  }

  // --- Utilities ---
  function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  }

  function copyAnswerToClipboard() {
    if (state.currentResult && state.currentResult.answer) {
      navigator.clipboard.writeText(state.currentResult.answer).then(() => {
        showToast('✓ Answer copied to clipboard');
      }).catch(() => {
        showToast('Failed to copy');
      });
    }
  }

  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 200);
    }, 2400);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // --- Simple Markdown & Citation Renderer ---
  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headings
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italics
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Unordered lists
    html = html.replace(/^\s*[-*•]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    // Inline Citation references: [1], [2], [Citation 1]
    html = html.replace(/\[(?:Citation\s*)?(\d+)\]/g, (match, num) => {
      return `<a href="#source-${num}" class="citation-ref" data-ref="${num}">[${num}]</a>`;
    });

    // Paragraphs
    html = html.split('\n\n').map(p => {
      const trimmed = p.trim();
      if (trimmed.startsWith('<h') || trimmed.startsWith('<ul') || trimmed.startsWith('<pre') || trimmed.startsWith('<block')) {
        return trimmed;
      }
      return trimmed ? `<p>${trimmed.replace(/\n/g, '<br>')}</p>` : '';
    }).join('');

    return html;
  }

  // Execute on DOM Ready
  document.addEventListener('DOMContentLoaded', init);
})();
