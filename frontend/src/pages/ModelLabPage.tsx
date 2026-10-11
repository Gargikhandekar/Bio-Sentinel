import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { EmptyState } from "../components/Primitives";
import { api } from "../services/api";

export function ModelLabPage() {
  const [rows, setRows] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    api.models().then((data: any) => {
      if (data.status) setStatus(data.status);
      setRows(data.models || []);
    }).catch(() => setStatus("Awaiting training run"));
  }, []);

  if (status && !rows.length) return <section className="p-8"><EmptyState message={status} /></section>;

  return (
    <section className="px-4 py-8 lg:px-10">
      <h1 className="text-2xl">MODEL LAB</h1>
      <div className="mt-6 overflow-x-auto rounded-xl border border-white/10">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-panel2 text-mute">
            <tr>
              {["Model", "Accuracy", "F1", "ROC-AUC", "PR-AUC", "Brier", "ECE"].map((h) => (
                <th key={h} className="px-3 py-2 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.id}
                className="cursor-pointer border-t border-white/10 hover:bg-white/5"
                onClick={() => {
                  api.model(row.id).then(setSelected);
                }}
              >
                <td className="mono px-3 py-2">{row.id}</td>
                <td className="mono px-3 py-2">{fmt(row.accuracy)}</td>
                <td className="mono px-3 py-2">{fmt(row.f1)}</td>
                <td className="mono px-3 py-2">{fmt(row.roc_auc)}</td>
                <td className="mono px-3 py-2">{fmt(row.pr_auc)}</td>
                <td className="mono px-3 py-2">{fmt(row.brier)}</td>
                <td className="mono px-3 py-2">{fmt(row.ece)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected?.metrics?.roc_curve && (
        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <Chart title="ROC" x={selected.metrics.roc_curve.fpr} y={selected.metrics.roc_curve.tpr} />
          <Chart title="PR" x={selected.metrics.pr_curve.recall} y={selected.metrics.pr_curve.precision} />
        </div>
      )}
    </section>
  );
}

function fmt(v: number | null | undefined) {
  return v == null ? "—" : v.toFixed(3);
}

function Chart({ title, x, y }: { title: string; x: number[]; y: number[] }) {
  return (
    <div className="rounded-xl border border-white/10 bg-panel p-3">
      <p className="mono text-xs text-mute">{title}</p>
      <ReactECharts
        style={{ height: 280 }}
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 16, top: 24, bottom: 32 },
          xAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#9AA6B2" } },
          yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#9AA6B2" } },
          series: [{ type: "line", data: x.map((v, i) => [v, y[i]]), showSymbol: false, lineStyle: { color: "#55E6D2" } }],
        }}
      />
    </div>
  );
}
