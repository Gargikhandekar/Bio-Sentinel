import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { EmptyState, Metric } from "../components/Primitives";
import { api } from "../services/api";

export function ResearchPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [ablation, setAblation] = useState<any>(null);
  const [calibration, setCalibration] = useState<any>(null);

  useEffect(() => {
    Promise.all([api.researchMetrics(), api.ablation(), api.calibration()])
      .then(([m, a, c]) => {
        setMetrics(m);
        setAblation(a);
        setCalibration(c);
      })
      .catch(() => setMetrics({ status: "Experiment not yet executed" }));
  }, []);

  if (!metrics || metrics.status) {
    return (
      <section className="p-8">
        <EmptyState message={metrics?.status || "Experiment not yet executed"} />
      </section>
    );
  }

  const diabetes = metrics.metrics?.diabetes || {};
  const acwe = diabetes.acwe_calibrated || {};

  return (
    <section className="px-4 py-8 lg:px-10">
      <p className="mono text-[11px] tracking-[0.25em] text-aqua">RESEARCH QUESTION</p>
      <h1 className="mt-2 max-w-3xl text-2xl md:text-3xl">
        Can adaptive model selection maintain reliable risk prediction under distribution shift?
      </h1>
      <p className="mt-4 max-w-3xl text-sm text-mute">
        Data → Baselines → Online learning → Drift → Dynamic ensemble → Calibration → Uncertainty → Ablation
      </p>
      <div className="mt-4 flex flex-wrap gap-3 text-sm">
        <Link className="text-aqua" to="/research/ablation">
          Ablation view
        </Link>
        <Link className="text-aqua" to="/research/calibration">
          Calibration view
        </Link>
      </div>
      {diabetes.chain_vs_independent && (
        <p className="mt-4 max-w-3xl text-sm text-mute">
          Independent best ROC-AUC {fmt(diabetes.chain_vs_independent.independent_best_roc_auc)} · classifier chain{" "}
          {fmt(diabetes.chain_vs_independent.classifier_chain_roc_auc)} · ACWE {fmt(diabetes.chain_vs_independent.acwe_roc_auc)} ·
          dependency-aware {fmt(diabetes.chain_vs_independent.dependency_aware_roc_auc)} · raw ECE{" "}
          {fmt(diabetes.chain_vs_independent.raw_ece)} · calibrated ECE {fmt(diabetes.chain_vs_independent.acwe_ece)}
        </p>
      )}
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="ACWE ROC-AUC" value={fmt(acwe.roc_auc)} />
        <Metric label="Brier" value={fmt(acwe.brier)} />
        <Metric label="ECE" value={fmt(acwe.ece)} />
        <Metric label="Test n" value={String(acwe.n ?? "—")} />
      </div>
      <h2 className="mt-10 text-lg">Ablation</h2>
      <div className="mt-3 overflow-x-auto rounded-xl border border-white/10">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-panel2 text-mute">
            <tr>
              <th className="px-3 py-2">Variant</th>
              <th className="px-3 py-2">ROC-AUC</th>
              <th className="px-3 py-2">Δ vs full</th>
              <th className="px-3 py-2">Brier</th>
            </tr>
          </thead>
          <tbody>
            {(ablation?.ablation || []).map((row: any) => (
              <tr key={row.variant} className="border-t border-white/10">
                <td className="px-3 py-2">{row.variant}</td>
                <td className="mono px-3 py-2">{fmt(row.roc_auc)}</td>
                <td className="mono px-3 py-2">{fmt(row.delta_vs_full)}</td>
                <td className="mono px-3 py-2">{fmt(row.brier)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-6 text-xs text-mute">
        Reliability bins (calibrated): {(calibration?.reliability?.calibrated || []).length} bins stored from the validation-safe isotonic fit.
      </p>
    </section>
  );
}

function fmt(v: number | null | undefined) {
  return v == null ? "—" : Number(v).toFixed(3);
}
