import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = 'http://localhost:8080';

const TEST_PROMPTS = [
  'What is r4r?',
  'What is Spring AI?',
  'When should I use RAG instead of fine-tuning?',
  'What is the role of Ollama in a Spring AI application?',
  'Explain dependency injection in Spring Boot.',
  'What is the recommended architecture for an enterprise Spring AI RAG application?',
  'Where is company data stored and what is the SLA?'
];

function App() {
  // Chat state
  const [q, setQ] = useState('');
  const [a, setA] = useState('');
  const [modelUsed, setModelUsed] = useState('');
  const [ragUsed, setRagUsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [enableRag, setEnableRag] = useState(true);

  // RAG upload state
  const [ragFile, setRagFile] = useState(null);
  const [ragStatus, setRagStatus] = useState('');
  const [ragLoading, setRagLoading] = useState(false);

  // Fine-tuning state
  const [ftFile, setFtFile] = useState(null);
  const [ftUploadStatus, setFtUploadStatus] = useState('');
  const [datasetFiles, setDatasetFiles] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('spring-ai.jsonl');
  const [ftModelName, setFtModelName] = useState('spring-ai-expert');
  const [ftBaseModel, setFtBaseModel] = useState('qwen2.5:1.5b');
  const [ftProcessing, setFtProcessing] = useState(false);
  const [ftResult, setFtResult] = useState(null);
  const [showLogs, setShowLogs] = useState(false);

  // Load models and dataset files on mount
  useEffect(() => {
    loadModels();
    loadDatasetFiles();
  }, []);

  async function loadModels() {
    try {
      const res = await fetch(`${API}/api/finetune/models`);
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data);
        if (data.length > 0 && !selectedModel) {
          setSelectedModel(data[0]);
        }
      }
    } catch (err) {
      console.warn('Could not load models from backend', err);
    }
  }

  async function loadDatasetFiles() {
    try {
      const res = await fetch(`${API}/api/finetune/files`);
      if (res.ok) {
        const files = await res.json();
        setDatasetFiles(files);
        if (files.length > 0 && !selectedDataset) {
          setSelectedDataset(files[0].name);
        }
      }
    } catch (err) {
      console.warn('Could not load dataset files', err);
    }
  }

  // Submit Chat Prompt
  async function ask(e, overridePrompt, overrideModel) {
    if (e && e.preventDefault) e.preventDefault();
    const promptToSend = (overridePrompt !== undefined ? overridePrompt : q).trim();
    if (!promptToSend) return;

    const modelToSend = overrideModel || selectedModel;
    setLoading(true);
    setA('');
    setModelUsed('');
    setRagUsed(false);

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: promptToSend,
          model: modelToSend,
          enableRag: enableRag
        })
      });
      const data = await res.json();
      setA(data.answer || data.message || 'No response returned.');
      setModelUsed(data.modelUsed || modelToSend || 'default');
      setRagUsed(Boolean(data.ragEnabled));
    } catch (err) {
      setA('Backend unavailable. Please ensure Spring Boot is running on port 8080.');
    } finally {
      setLoading(false);
    }
  }

  function handlePromptClick(promptText) {
    setQ(promptText);
    ask(null, promptText);
  }

  // Upload RAG document to PGVector
  async function uploadRag() {
    if (!ragFile) return;
    setRagLoading(true);
    setRagStatus('Uploading and indexing document into PGVector…');
    const f = new FormData();
    f.append('file', ragFile);
    try {
      const res = await fetch(`${API}/api/rag/file`, { method: 'POST', body: f });
      const data = await res.json();
      setRagStatus(`✅ Indexed ${data.documentsAdded} document chunk(s) into PGVector.`);
      setRagFile(null);
    } catch (err) {
      setRagStatus('❌ Document indexing failed.');
    } finally {
      setRagLoading(false);
    }
  }

  // Upload dataset file to fine-tune/dataset/
  async function uploadFineTuneFile() {
    if (!ftFile) return;
    setFtUploadStatus('Uploading dataset file to fine-tune/dataset/…');
    const f = new FormData();
    f.append('file', ftFile);
    try {
      const res = await fetch(`${API}/api/finetune/upload`, { method: 'POST', body: f });
      const data = await res.json();
      if (res.ok) {
        setFtUploadStatus(`✅ Uploaded ${data.fileName} (${data.sampleCount} samples, ${(data.size / 1024).toFixed(1)} KB)`);
        setFtFile(null);
        await loadDatasetFiles();
        setSelectedDataset(data.fileName);
      } else {
        setFtUploadStatus(`❌ Upload failed: ${data.message || 'Server error'}`);
      }
    } catch (err) {
      setFtUploadStatus('❌ Upload failed. Ensure Spring Boot is running.');
    }
  }

  // Trigger fine-tuning process
  async function processFineTuning() {
    if (!selectedDataset) {
      setFtResult({ success: false, message: 'Please select or upload a dataset file first.' });
      return;
    }
    setFtProcessing(true);
    setFtResult(null);

    try {
      const res = await fetch(`${API}/api/finetune/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetFile: selectedDataset,
          modelName: ftModelName,
          baseModel: ftBaseModel,
          temperature: 0.1
        })
      });
      const data = await res.json();
      setFtResult(data);

      if (data.success) {
        await loadModels();
        setSelectedModel(data.modelName);
      }
    } catch (err) {
      setFtResult({
        success: false,
        message: 'Fine-tuning process failed to connect to backend: ' + err.message,
        logs: err.stack
      });
    } finally {
      setFtProcessing(false);
    }
  }

// Trigger fine-tuning process
  async function processLoraFineTuning() {
    if (!selectedDataset) {
      setFtResult({ success: false, message: 'Please select or upload a dataset file first.' });
      return;
    }
    setFtProcessing(true);
    setFtResult(null);

    try {
      const res = await fetch(`${API}/api/lorafinetune/loraprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetFile: selectedDataset,
          modelName: ftModelName,
          baseModel: ftBaseModel,
          temperature: 0.1
        })
      });
      const data = await res.json();
      setFtResult(data);

      if (data.success) {
        await loadModels();
        setSelectedModel(data.modelName);
      }
    } catch (err) {
      setFtResult({
        success: false,
        message: 'Fine-tuning process failed to connect to backend: ' + err.message,
        logs: err.stack
      });
    } finally {
      setFtProcessing(false);
    }
  }
  return (
    <div className="app">
      <header>
        <div>
          <div className="eyebrow">SPRING BOOT 4 • SPRING AI • OLLAMA • PGVECTOR</div>
          <h1>AI Studio & Local RAG</h1>
          <p>Fine-Tune Custom Models • Index Knowledge in PGVector • Test via Local Inference</p>
        </div>
        <div className="header-badges">
          <span className="pill green">● Ollama Inference Active</span>
          <span className="pill blue">● PGVector Connected</span>
        </div>
      </header>

      <main>
        {/* Left Column: Fine-Tuning & Knowledge Base */}
        <div className="column">
          {/* Fine-Tuning Studio Card */}
          <section className="card highlight-card">
            <div className="card-header">
              <div className="card-tag">FINE-TUNING OPERATIONS</div>
              <h2>Fine-Tune via GUI</h2>
            </div>
            <p>Upload training datasets to <code>fine-tune/dataset/</code> and process them into custom Ollama models.</p>

            {/* Step 1: Upload File */}
            <div className="section-block">
              <label className="section-label">1. Upload Training Dataset (.jsonl, .json, .csv, .txt)</label>
              <div className="upload-row">
                <input
                  type="file"
                  accept=".jsonl,.json,.csv,.txt"
                  onChange={e => setFtFile(e.target.files[0])}
                />
                <button
                  type="button"
                  className="btn-primary"
                  onClick={uploadFineTuneFile}
                  disabled={!ftFile}
                >
                  Submit File
                </button>
              </div>
              {ftUploadStatus && (
                <div className={`status-msg ${ftUploadStatus.startsWith('✅') ? 'success' : ftUploadStatus.startsWith('❌') ? 'error' : ''}`}>
                  {ftUploadStatus}
                </div>
              )}
            </div>

            {/* Step 2: Configuration & Process */}
            <div className="section-block">
              <label className="section-label">2. Configure & Process Model</label>
              <div className="form-grid">
                <div className="form-group">
                  <label>Select Dataset File</label>
                  <select
                    value={selectedDataset}
                    onChange={e => setSelectedDataset(e.target.value)}
                  >
                    {datasetFiles.map(f => (
                      <option key={f.name} value={f.name}>
                        {f.name} ({f.sampleCount} lines, {(f.size / 1024).toFixed(1)} KB)
                      </option>
                    ))}
                    {datasetFiles.length === 0 && <option value="spring-ai-1.jsonl">spring-ai-1.jsonl</option>}
                  </select>
                </div>

                <div className="form-group">
                  <label>Target Model Name</label>
                  <input
                    type="text"
                    value={ftModelName}
                    onChange={e => setFtModelName(e.target.value)}
                    placeholder="e.g. spring-ai-expert"
                  />
                </div>

                <div className="form-group">
                  <label>Base Model</label>
                  <select
                    value={ftBaseModel}
                    onChange={e => setFtBaseModel(e.target.value)}
                  >
                    {availableModels.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                    {availableModels.length === 0 && <option value="qwen2.5:1.5b">qwen2.5:1.5b</option>}
                  </select>
                </div>
              </div>

              <div className="process-action">
                <button
                  type="button"
                  className="btn-accent process-btn"
                  onClick={processFineTuning}
                  disabled={ftProcessing}
                >
                  {ftProcessing ? (
                    <span className="spinner-text">⏳ Processing Fine-Tuning…</span>
                  ) : (
                    '⚡ Process & Build Fine-Tuned Model'
                  )}
                </button>
              </div>
              <div className="process-action">
                              <button
                                type="button"
                                className="btn-accent process-btn"
                                onClick={processLoraFineTuning}
                                disabled={ftProcessing}
                              >
                                {ftProcessing ? (
                                  <span className="spinner-text">⏳ Processing LoRa/QLoRa Fine-Tuning…</span>
                                ) : (
                                  '⚡ Process Via LoRA/QLoRA & Build Fine-Tuned Model'
                                )}
                              </button>
               </div>
            </div>

            {/* Step 3: Result Message (Success / Failed) */}
            {ftResult && (
              <div className={`result-banner ${ftResult.success ? 'banner-success' : 'banner-error'}`}>
                <div className="banner-title">
                  {ftResult.success ? '✅ SUCCESS: Model Ready' : '❌ FAILED: Processing Error'}
                </div>
                <div className="banner-content">{ftResult.message}</div>
                {ftResult.logs && (
                  <div className="logs-container">
                    <button
                      type="button"
                      className="btn-link"
                      onClick={() => setShowLogs(!showLogs)}
                    >
                      {showLogs ? 'Hide Processing Logs ▲' : 'Show Processing Logs ▼'}
                    </button>
                    {showLogs && <pre className="logs-pre">{ftResult.logs}</pre>}
                  </div>
                )}
                {ftResult.success && (
                  <div className="banner-tip">
                    👉 Model <strong>{ftResult.modelName}</strong> is now compiled and ready for testing!
                  </div>
                )}
              </div>
            )}
          </section>

          {/* RAG Knowledge Base Ingestion Card */}
          <section className="card">
            <div className="card-header">
              <div className="card-tag">RAG STORAGE (PGVECTOR)</div>
              <h2>Knowledge Base Ingestion</h2>
            </div>
            <p>Upload enterprise documentation (.md, .json, .csv, .txt) to embed and store in PGVector.</p>
            <div className="upload-row">
              <input
                type="file"
                accept=".txt,.md,.json,.csv"
                onChange={e => setRagFile(e.target.files[0])}
              />
              <button
                type="button"
                className="btn-secondary"
                onClick={uploadRag}
                disabled={!ragFile || ragLoading}
              >
                {ragLoading ? 'Indexing…' : 'Index Document'}
              </button>
            </div>
            {ragStatus && (
              <div className={`status-msg ${ragStatus.startsWith('✅') ? 'success' : ragStatus.startsWith('❌') ? 'error' : ''}`}>
                {ragStatus}
              </div>
            )}
          </section>
        </div>

        {/* Right Column: Chatbot Testing with Test Prompts */}
        <div className="column">
          <section className="card chat-card">
            <div className="card-header flex-between">
              <div>
                <div className="card-tag">INTERACTIVE TEST BENCH</div>
                <h2>Test Chatbot on Same Page</h2>
              </div>
              <div className="model-controls">
                <label className="rag-toggle">
                  <input
                    type="checkbox"
                    checked={enableRag}
                    onChange={e => setEnableRag(e.target.checked)}
                  />
                  <span>RAG Context</span>
                </label>
                <div className="model-selector-wrap">
                  <select
                    value={selectedModel}
                    onChange={e => setSelectedModel(e.target.value)}
                    className="model-select"
                  >
                    {availableModels.map(m => (
                      <option key={m} value={m}>
                        {m.includes('expert') ? `⭐ ${m}` : m}
                      </option>
                    ))}
                    {availableModels.length === 0 && <option value="qwen2.5:1.5b">qwen2.5:1.5b</option>}
                  </select>
                </div>
              </div>
            </div>

            {/* Test Prompts Section */}
            <div className="test-prompts-section">
              <div className="prompts-title">
                <span>💡 Click a Test Prompt to run:</span>
              </div>
              <div className="prompt-chips">
                {TEST_PROMPTS.map((promptText, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="chip-btn"
                    onClick={() => handlePromptClick(promptText)}
                    disabled={loading}
                  >
                    💬 {promptText}
                  </button>
                ))}
              </div>
            </div>

            {/* Chat Form */}
            <form onSubmit={ask} className="chat-form">
              <textarea
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder="Ask questions about your fine-tuned model or indexed knowledge base..."
                rows={4}
              />
              <div className="chat-actions">
                <button
                  type="submit"
                  className="btn-primary submit-btn"
                  disabled={loading || !q.trim()}
                >
                  {loading ? 'Thinking…' : 'Ask AI Model'}
                </button>
                <div className="active-meta">
                  {selectedModel && (
                    <span className="active-model-tag">
                      Model: <strong>{selectedModel}</strong>
                    </span>
                  )}
                  <span className={`rag-indicator ${enableRag ? 'on' : 'off'}`}>
                    RAG: {enableRag ? 'ON' : 'OFF'}
                  </span>
                </div>
              </div>
            </form>

            {/* AI Response Display */}
            {a && (
              <div className="answer-box">
                <div className="answer-header">
                  <div className="label">AI RESPONSE</div>
                  <div className="answer-badges">
                    {modelUsed && (
                      <span className="badge-model">
                        Model: {modelUsed}
                      </span>
                    )}
                    <span className={`badge-rag ${ragUsed ? 'active' : 'inactive'}`}>
                      {ragUsed ? 'RAG Augmented' : 'Fine-Tuned Direct'}
                    </span>
                  </div>
                </div>
                <div className="answer-body">{a}</div>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer>
        Spring Boot 4 • Spring AI 2.0.2 • Ollama Local Inference • PostgreSQL + PGVector
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
