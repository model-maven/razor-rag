// Talks to the FastAPI backend. The /api prefix is proxied by Vite in dev
// and served by the same FastAPI server in production.

export async function fetchHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}

/**
 * Send a chat message and stream the response.
 * Calls the handlers as newline-delimited JSON events arrive:
 *   onSources(sources[]), onToken(text), onError(detail)
 */
export async function streamChat({ message, history, onSources, onToken, onError }) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok || !res.body) {
    onError(`Server error (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep the incomplete tail for the next chunk

    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      if (event.type === "sources") onSources(event.sources);
      else if (event.type === "token") onToken(event.text);
      else if (event.type === "error") onError(event.detail);
    }
  }
}
