import { useEffect, useRef, useState } from "react";
import { fetchHealth, streamChat } from "./api.js";

const SUGGESTIONS = [
  "What are these documents about?",
  "Summarize the key points",
  "What conclusions are drawn?",
];

function SourceCard({ s }) {
  const [open, setOpen] = useState(false);
  return (
    <button className={"source-card" + (open ? " open" : "")} onClick={() => setOpen(!open)}>
      <span className="source-tab">p. {s.page}</span>
      <span className="source-body">
        <span className="source-name">{s.source}</span>
        <span className="source-score">relevance {(s.score * 100).toFixed(0)}%</span>
        {open && <span className="source-excerpt">{s.text.slice(0, 320)}{s.text.length > 320 ? "…" : ""}</span>}
      </span>
    </button>
  );
}

function Message({ m, streaming }) {
  if (m.role === "user") {
    return <div className="msg-user">{m.content}</div>;
  }
  return (
    <div className="msg-assistant">
      <div className="answer">
        {m.content}
        {streaming && <span className="caret" />}
      </div>
      {m.sources?.length > 0 && (
        <div className="sources">
          <div className="sources-label">Sources</div>
          <div className="sources-row">
            {m.sources.map((s, i) => (
              <SourceCard key={i} s={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);

    // history for the API: previous turns, text only
    const history = messages
      .filter((m) => m.content)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-6);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [] },
    ]);

    const update = (fn) =>
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });

    try {
      await streamChat({
        message: question,
        history,
        onSources: (sources) => update((m) => ({ ...m, sources })),
        onToken: (t) => update((m) => ({ ...m, content: m.content + t })),
        onError: (detail) =>
          update((m) => ({ ...m, content: m.content || `Something went wrong: ${detail}` })),
      });
    } catch {
      update((m) => ({ ...m, content: m.content || "Couldn't reach the server. Is the backend running?" }));
    } finally {
      setBusy(false);
    }
  }

  const statusText =
    health == null
      ? "Connecting…"
      : health.status === "ok"
      ? `Connected · ${health.vectors.toLocaleString()} passages indexed`
      : health.status === "no_index"
      ? "No documents indexed yet — run ingest.py"
      : "Backend offline — start the API server";

  return (
    <div className="page">
      <header className="topbar">
        <div className="wordmark">
          <span className="wordmark-box">DA</span> Document Assistant
        </div>
        <div className={"status " + (health?.status === "ok" ? "ok" : "warn")}>
          <span className="dot" /> {statusText}
        </div>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <div className="empty">
            <h1>Ask the documents.</h1>
            <p>
              Answers come straight from the loaded files, with page-level citations you can
              check. If the documents don't cover it, you'll be told — no guessing.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <Message
            key={i}
            m={m}
            streaming={busy && i === messages.length - 1 && m.role === "assistant"}
          />
        ))}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the documents…"
            aria-label="Your question"
            maxLength={2000}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>
      </footer>
    </div>
  );
}
