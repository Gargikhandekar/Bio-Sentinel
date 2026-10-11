import { useEffect, useState } from "react";
import { api } from "../services/api";

export function SystemPage() {
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setError("Unable to reach the model service."));
  }, []);
  return (
    <section className="px-4 py-8 lg:px-10">
      <h1 className="text-2xl">SYSTEM STATUS</h1>
      {error && <p className="mt-4 text-crit">{error}</p>}
      <ul className="mt-6 max-w-lg space-y-3">
        {Object.entries(health?.services || {}).map(([name, status]) => (
          <li key={name} className="flex items-center justify-between rounded-lg border border-white/10 bg-panel px-4 py-3">
            <span className="mono text-xs tracking-[0.14em]">{name.replace(/_/g, " ")}</span>
            <span className={`mono text-xs ${status === "OPERATIONAL" ? "text-aqua" : "text-warn"}`}>{String(status)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
