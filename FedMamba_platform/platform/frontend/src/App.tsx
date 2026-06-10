import { useEffect, useState } from "react";

interface HealthResponse {
  status: string;
  version: string;
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/health")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: HealthResponse) => setHealth(data))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div style={{ fontFamily: "system-ui", padding: "2rem" }}>
      <h1>FedMamba-SALT Clinical Platform</h1>
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
      {health && (
        <p>
          Status: <strong>{health.status}</strong> — Version:{" "}
          <strong>{health.version}</strong>
        </p>
      )}
      {!health && !error && <p>Connecting to backend...</p>}
    </div>
  );
}

export default App;
