import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { EmptyState } from "../components/Primitives";
import { api } from "../services/api";

export function AblationPage() {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    api.ablation().then(setData).catch(() => setData({ status: "Experiment not yet executed" }));
  }, []);
  if (!data || data.status) {
    return (
      <section className="p-8">
        <EmptyState message={data?.status || "Experiment not yet executed"} />
      </section>
    );
  }
  const rows = data.ablation || [];
  const full = rows.find((r: any) => r.variant === "full");
  return (
    <section className="px-4 py-8 lg:px-10">
      <p className="mono text-[11px] tracking-[0.25em] text-aqua">ABLATION</p>
      <h1 className="mt-2 text-2xl">Which components move the metric?</h1>
      <p className="mt-4 font-mono text-xl text-aqua">FULL SYSTEM {full ? Number(full.roc_auc).toFixed(3) : "—"}</p>
      <ul className="mt-6 max-w-xl space-y-2 text-sm">
        {rows
          .filter((r: any) => r.variant !== "full")
          .map((row: any) => (
            <li key={row.variant} className="flex justify-between border-b border-white/10 py-2">
              <span>remove {row.variant.replace("no_", "").replaceAll("_", " ")}</span>
              <span className="mono text-mute">{row.delta_vs_full == null ? "—" : Number(row.delta_vs_full).toFixed(3)}</span>
            </li>
          ))}
      </ul>
      <Link to="/research" className="mt-8 inline-block text-sm text-aqua">
        Back to research console
      </Link>
    </section>
  );
}
