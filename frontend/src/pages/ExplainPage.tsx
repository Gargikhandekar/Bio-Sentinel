import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "../components/Primitives";
import { api } from "../services/api";

export function ExplainPage() {
  const [payload, setPayload] = useState<any>(null);
  useEffect(() => {
    api
      .researchMetrics()
      .then(setPayload)
      .catch(() => setPayload({ status: "Awaiting training run" }));
  }, []);
  const points = payload?.feature_space?.points || [];
  const scaled = useMemo(() => {
    if (!points.length) return [];
    const xs = points.map((p: any) => p.x);
    const ys = points.map((p: any) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;
    return points.map((p: any) => ({
      x: ((p.x - minX) / dx) * 100,
      y: ((p.y - minY) / dy) * 100,
      label: p.label,
    }));
  }, [points]);
  if (!payload || payload.status) {
    return (
      <section className="p-8">
        <EmptyState message={payload?.status || "Awaiting training run"} />
      </section>
    );
  }
  const globalRows = payload.explainability?.global || [];
  return (
    <section className="px-4 py-8 lg:px-10">
      <h1 className="text-2xl">WHY DID THE SYSTEM DECIDE THIS?</h1>
      <p className="mt-3 max-w-2xl rounded-md border border-white/10 bg-panel px-4 py-3 text-sm text-mute">
        Model contribution does not imply causation. Model contribution ≠ causal medical effect.
      </p>
      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <article className="rounded-xl border border-white/10 bg-panel p-5">
          <p className="mono text-[11px] tracking-[0.2em] text-aqua">GLOBAL | MEAN |SHAP|</p>
          <ul className="mt-4 space-y-2">
            {globalRows.slice(0, 12).map((row: any) => (
              <li key={row.feature} className="flex justify-between text-sm">
                <span className="truncate pr-4">{row.feature}</span>
                <span className="mono text-aqua">{row.mean_abs_shap?.toFixed(4)}</span>
              </li>
            ))}
            {!globalRows.length && <li className="text-mute">No SHAP values stored yet.</li>}
          </ul>
        </article>
        <article className="rounded-xl border border-white/10 bg-panel p-5">
          <p className="mono text-[11px] tracking-[0.2em] text-aqua">PERMUTATION IMPORTANCE</p>
          <ul className="mt-4 space-y-2">
            {(payload.permutation_importance || []).slice(0, 12).map((row: any) => (
              <li key={row.feature} className="flex justify-between text-sm">
                <span className="truncate pr-4">{row.feature}</span>
                <span className="mono">{row.importance?.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </article>
      </div>
      {scaled.length > 0 && (
        <article className="mt-8 rounded-xl border border-white/10 bg-panel p-5">
          <p className="mono text-[11px] tracking-[0.2em] text-aqua">FEATURE SPACE ({payload.feature_space.method})</p>
          <svg viewBox="0 0 100 60" className="mt-4 h-80 w-full bg-ink" role="img" aria-label="Feature space embedding of held-out respondents">
            {scaled.slice(0, 800).map((p: any, i: number) => (
              <circle key={i} cx={p.x} cy={(p.y / 100) * 60} r="0.45" fill={p.label ? "#FF6B7A" : "#55E6D2"} opacity="0.75" />
            ))}
          </svg>
          <p className="mt-2 text-xs text-mute">Teal = negative diabetes label, coral = positive. Coordinates are an offline embedding of the test partition.</p>
        </article>
      )}
    </section>
  );
}
