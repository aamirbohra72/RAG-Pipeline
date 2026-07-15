"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "rag_access_token";
const USER_KEY = "rag_user";

type User = { id: string; email: string };

type Document = {
  doc_id: string;
  filename: string;
  chunks: number;
};

type Source = {
  filename: string;
  page: number;
  snippet: string;
  score?: number | null;
  rerank_score?: number | null;
  vector_score?: number | null;
  lexical_score?: number | null;
};

type ChatTurn = {
  question: string;
  answer: string;
  sources: Source[];
  streaming?: boolean;
};

function authHeaders(token: string, json = false): HeadersInit {
  const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedToken = localStorage.getItem(TOKEN_KEY);
    const savedUser = localStorage.getItem(USER_KEY);
    if (savedToken && savedUser) {
      setToken(savedToken);
      try {
        setUser(JSON.parse(savedUser));
      } catch {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
      }
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken(null);
    setUser(null);
    setDocuments([]);
    setHistory([]);
    setError("");
  }, []);

  const loadDocuments = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_URL}/documents`, {
        headers: authHeaders(token),
      });
      if (res.status === 401) {
        logout();
        return;
      }
      const data = await res.json();
      setDocuments(data.documents || []);
      setError("");
    } catch {
      setError("Could not reach backend. Is it running on " + API_URL + "?");
    }
  }, [token, logout]);

  useEffect(() => {
    if (token) loadDocuments();
  }, [token, loadDocuments]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const handleAuth = async (e: FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setError("");
    try {
      const path = authMode === "login" ? "/auth/login" : "/auth/register";
      const res = await fetch(`${API_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        const message =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(", ")
              : "Auth failed";
        throw new Error(message);
      }

      localStorage.setItem(TOKEN_KEY, data.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      setToken(data.access_token);
      setUser(data.user);
      setPassword("");
      setHistory([]);
    } catch (err) {
      const msg = (err as Error).message;
      setError(typeof msg === "string" ? msg : "Auth failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleUpload = async (fileList: FileList | File[] | null) => {
    if (!token || !fileList || (fileList as FileList).length === 0) return;
    setUploading(true);
    setError("");
    const formData = new FormData();
    Array.from(fileList).forEach((file) => formData.append("files", file));

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        headers: authHeaders(token),
        body: formData,
      });
      if (res.status === 401) {
        logout();
        return;
      }
      if (!res.ok) throw new Error(await res.text());
      await loadDocuments();
    } catch (err) {
      setError("Upload failed: " + (err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!token) return;
    const res = await fetch(`${API_URL}/documents/${docId}`, {
      method: "DELETE",
      headers: authHeaders(token),
    });
    if (res.status === 401) {
      logout();
      return;
    }
    await loadDocuments();
  };

  const handleAsk = async () => {
    if (!token) return;
    const q = question.trim();
    if (!q || asking) return;

    setAsking(true);
    setError("");
    setQuestion("");

    const priorHistory = history.flatMap((t) => [
      { role: "user", content: t.question },
      { role: "assistant", content: t.answer },
    ]);

    setHistory((prev) => [...prev, { question: q, answer: "", sources: [], streaming: true }]);

    try {
      const res = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        headers: authHeaders(token, true),
        body: JSON.stringify({ question: q, history: priorHistory.slice(-6) }),
      });

      if (res.status === 401) {
        logout();
        throw new Error("Session expired — please log in again");
      }
      if (!res.ok || !res.body) throw new Error(await res.text());

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let sources: Source[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));

          if (event.type === "sources") {
            sources = event.sources || [];
            setHistory((prev) => {
              const next = [...prev];
              next[next.length - 1] = { ...next[next.length - 1], sources };
              return next;
            });
          } else if (event.type === "token") {
            answer += event.content;
            setHistory((prev) => {
              const next = [...prev];
              next[next.length - 1] = {
                ...next[next.length - 1],
                answer,
                sources,
                streaming: true,
              };
              return next;
            });
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }

      setHistory((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          ...next[next.length - 1],
          answer,
          sources,
          streaming: false,
        };
        return next;
      });
    } catch (err) {
      setError("Query failed: " + (err as Error).message);
      setHistory((prev) => prev.slice(0, -1));
    } finally {
      setAsking(false);
    }
  };

  if (!token || !user) {
    return (
      <main className="max-w-md mx-auto px-4 py-16">
        <header className="mb-8 text-center">
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">Senior RAG</p>
          <h1 className="text-2xl font-semibold text-slate-900 mb-1">Chat with your PDFs</h1>
          <p className="text-sm text-slate-500">Sign in to access your private document library.</p>
        </header>

        {error && (
          <div className="mb-4 text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-3 py-2">
            {typeof error === "string" ? error : "Something went wrong"}
          </div>
        )}

        <form onSubmit={handleAuth} className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
          <div className="flex gap-2 mb-2">
            <button
              type="button"
              onClick={() => setAuthMode("login")}
              className={`flex-1 text-sm py-1.5 rounded-md ${
                authMode === "login" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"
              }`}
            >
              Log in
            </button>
            <button
              type="button"
              onClick={() => setAuthMode("register")}
              className={`flex-1 text-sm py-1.5 rounded-md ${
                authMode === "register" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"
              }`}
            >
              Register
            </button>
          </div>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm"
          />
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password (min 8 chars)"
            className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={authLoading}
            className="w-full bg-slate-900 text-white text-sm py-2 rounded-md disabled:opacity-50"
          >
            {authLoading ? "Please wait..." : authMode === "login" ? "Log in" : "Create account"}
          </button>
        </form>
      </main>
    );
  }

  return (
    <main className="max-w-3xl mx-auto px-4 py-10">
      <header className="mb-8 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">Senior RAG</p>
          <h1 className="text-2xl font-semibold text-slate-900 mb-1">Chat with your PDFs</h1>
          <p className="text-sm text-slate-500">
            Your library only — hybrid retrieval + streamed answers.
          </p>
        </div>
        <div className="text-right shrink-0">
          <p className="text-xs text-slate-500 mb-1">{user.email}</p>
          <button onClick={logout} className="text-xs text-slate-700 underline hover:text-slate-900">
            Log out
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-3 py-2">
          {error}
        </div>
      )}

      <section
        className={`mb-8 rounded-lg border-2 border-dashed py-10 text-center transition-colors ${
          dragOver ? "border-slate-700 bg-slate-50" : "border-slate-300"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleUpload(e.dataTransfer.files);
        }}
      >
        <label className="cursor-pointer">
          <span className="text-sm text-slate-600">
            {uploading ? "Uploading & indexing..." : "Drop PDFs here, or click to browse"}
          </span>
          <input
            type="file"
            accept="application/pdf"
            multiple
            className="hidden"
            onChange={(e) => handleUpload(e.target.files)}
            disabled={uploading}
          />
        </label>
      </section>

      <section className="mb-10">
        <h2 className="text-sm font-medium text-slate-700 mb-2">
          Your documents ({documents.length})
        </h2>
        {documents.length === 0 ? (
          <p className="text-sm text-slate-400">No documents in your library yet.</p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li
                key={doc.doc_id}
                className="flex items-center justify-between text-sm bg-white border border-slate-200 rounded-md px-3 py-2"
              >
                <span>
                  {doc.filename}{" "}
                  <span className="text-slate-400">· {doc.chunks} chunks</span>
                </span>
                <button
                  onClick={() => handleDelete(doc.doc_id)}
                  className="text-red-500 hover:text-red-700 text-xs"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="text-sm font-medium text-slate-700 mb-2">Ask a question</h2>
        <div className="flex gap-2 mb-6">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder="What does the document say about..."
            className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400"
          />
          <button
            onClick={handleAsk}
            disabled={asking}
            className="bg-slate-900 text-white text-sm px-4 py-2 rounded-md disabled:opacity-50"
          >
            {asking ? "Thinking..." : "Ask"}
          </button>
        </div>

        <div className="space-y-6">
          {history.map((turn, i) => (
            <div key={i} className="bg-white border border-slate-200 rounded-lg p-4">
              <p className="text-sm font-medium text-slate-800 mb-2">Q: {turn.question}</p>
              <p className="text-sm text-slate-700 whitespace-pre-wrap mb-3">
                {turn.answer || (turn.streaming ? "…" : "")}
                {turn.streaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-slate-400 animate-pulse align-middle" />
                )}
              </p>
              {turn.sources.length > 0 && (
                <div className="text-xs text-slate-500 space-y-1 border-t border-slate-100 pt-2">
                  <p className="font-medium text-slate-600">Sources</p>
                  {turn.sources.map((s, j) => (
                    <p key={j}>
                      {s.filename} (page {s.page})
                      {typeof s.rerank_score === "number"
                        ? ` · rerank ${s.rerank_score.toFixed(3)}`
                        : typeof s.score === "number"
                          ? ` · score ${s.score.toFixed(3)}`
                          : ""}
                      {" — "}
                      &ldquo;{s.snippet}…&rdquo;
                    </p>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </section>
    </main>
  );
}
