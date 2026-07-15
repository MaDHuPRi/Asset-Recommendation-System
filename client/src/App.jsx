import { useEffect, useState, useCallback } from "react";
import RangeIndicator from "./RangeIndicator";
import "./tokens.css";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Fallback sample data so the dashboard still renders something coherent
// if the FastAPI backend (Day 5) isn't running -- e.g. when just reviewing
// the client in isolation. Shape matches the real API response.
const FALLBACK_USERS = [
  { user_id: 1, age: 41, risk_tolerance: 7.2, portfolio_value: 48210 },
  { user_id: 2, age: 55, risk_tolerance: 3.4, portfolio_value: 121400 },
  { user_id: 3, age: 29, risk_tolerance: 8.9, portfolio_value: 15200 },
];

function fallbackRecommendation(userId) {
  const seed = userId % 5;
  const estimate = [1.8, -0.6, 0.3, -1.4, 2.4][seed];
  const spread = 0.9 + (userId % 3) * 0.3;
  return {
    user_id: userId,
    recommended_action: estimate > 0 ? "Increase equity exposure" : "Maintain current allocation",
    estimated_uplift_pct: estimate,
    confidence_interval_90: [estimate - spread, estimate + spread],
    rationale: `Estimated causal effect of this recommendation on your 30-day portfolio return is ${estimate > 0 ? "+" : ""}${estimate.toFixed(2)}pp, based on your risk profile and current market conditions. (Offline sample data -- backend not connected.)`,
  };
}

export default function App() {
  const [users, setUsers] = useState(FALLBACK_USERS);
  const [selectedId, setSelectedId] = useState(FALLBACK_USERS[0].user_id);
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendConnected, setBackendConnected] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/users?limit=20`)
      .then((r) => {
        if (!r.ok) throw new Error("bad response");
        return r.json();
      })
      .then((data) => {
        setUsers(data);
        setBackendConnected(true);
      })
      .catch(() => setBackendConnected(false));
  }, []);

  const loadRecommendation = useCallback((userId) => {
    setLoading(true);
    fetch(`${API_BASE}/recommend/${userId}`)
      .then((r) => {
        if (!r.ok) throw new Error("bad response");
        return r.json();
      })
      .then((data) => {
        setRecommendation(data);
        setBackendConnected(true);
      })
      .catch(() => {
        setRecommendation(fallbackRecommendation(userId));
        setBackendConnected(false);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadRecommendation(selectedId);
  }, [selectedId, loadRecommendation]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <p className="eyebrow">Sample users</p>
          <h1 className="brand">Causal Portfolio Engine</h1>
        </div>
        <ul className="user-list">
          {users.map((u) => (
            <li key={u.user_id}>
              <button
                className={`user-row ${u.user_id === selectedId ? "user-row--active" : ""}`}
                onClick={() => setSelectedId(u.user_id)}
              >
                <span className="user-row-id">#{u.user_id}</span>
                <span className="user-row-meta">
                  risk {u.risk_tolerance?.toFixed?.(1) ?? u.risk_tolerance} · age {u.age}
                </span>
              </button>
            </li>
          ))}
        </ul>
        {!backendConnected && (
          <p className="backend-warning">
            API not reachable at {API_BASE} — showing offline sample data.
            Run <code>uvicorn src.serving.app:app --reload</code> from the repo root.
          </p>
        )}
      </aside>

      <main className="detail-panel">
        {loading && <p className="loading">Loading recommendation…</p>}
        {!loading && recommendation && (
          <>
            <p className="eyebrow">Recommendation for user #{recommendation.user_id}</p>
            <h2 className="recommendation-headline">{recommendation.recommended_action}</h2>

            <RangeIndicator
              estimate={recommendation.estimated_uplift_pct}
              ciLow={recommendation.confidence_interval_90[0]}
              ciHigh={recommendation.confidence_interval_90[1]}
            />
            <p className="range-caption">
              estimated causal uplift, 30-day forward return (90% confidence interval)
            </p>

            <div className="rationale-card">
              <p className="rationale-label">Why this recommendation</p>
              <p className="rationale-text">{recommendation.rationale}</p>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
