import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { EmptyState, Metric } from "../components/Primitives";
import { api } from "../services/api";

export function CalibrationPage() {
  const [data, setData] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  useEffect(() => {
    Promise.all([api.calibration(), api.researchMetrics()])
      .then(([c, m]) => {
        setData(c);
        setMetrics(m);
      })
      .catch(() => setData({ status: "Experiment not yet executed" }));
  }, []);
  if (!data || data.status) {
    return (
      <section className="p-8">
        <EmptyState message={data?.status || "Experiment not yet executed"} />
      </section>
    );
  }
  const cal = data.reliability?.calibrated || [];
  const raw = data.reliability?.raw || [];
  const diabetes = metrics?.metrics?.diabetes || {};
  return (
    <section className="px-4 py-8 lg:px-10">
      <h1 className="text-2xl">CALIBRATION</h1>
      <p className="mt-2 max-w-2xl text-sm text-mute">
        Reliability diagrams use validation-fitted isotonic mapping applied to the untouched test set. RQ6 asks whether
        calibration and uncertainty quality improve, not only AUROC.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Metric label="ACWE Brier" value={fmt(diabetes.acwe_calibrated?.brier)} />
        <Metric label="ACWE ECE" value={fmt(diabetes.acwe_calibrated?.ece)} />
        <Metric label="Raw ECE" value={fmt(diabetes.acwe_raw?.ece)} />
      </div>
      <div className="mt-6 rounded-xl border border-white/10 bg-panel p-3">
        <ReactECharts
          aria-label="Reliability diagram comparing raw and calibrated probabilities"
          style={{ height: 340 }}
          option={{
            backgroundColor: "transparent",
            legend: { textStyle: { color: "#9AA6B2" } },
            xAxis: { type: "value", min: 0, max: 1, name: "predicted", axisLabel: { color: "#9AA6B2" } },
            yAxis: { type: "value", min: 0, max: 1, name: "observed", axisLabel: { color: "#9AA6B2" } },
            series: [
              {
                name: "raw",
                type: "scatter",
                data: raw.filter((r: any) => r.mean_predicted != null).map((r: any) => [r.mean_predicted, r.mean_observed]),
                itemStyle: { color: "#FF6B7A" },
              },
              {
                name: "calibrated",
                type: "scatter",
                data: cal.filter((r: any) => r.mean_predicted != null).map((r: any) => [r.mean_predicted, r.mean_observed]),
                itemStyle: { color: "#55E6D2" },
              },
              { name: "perfect", type: "line", data: [[0, 0], [1, 1]], lineStyle: { color: "#9AA6B2", type: "dashed" }, showSymbol: false },
            ],
          }}
        />
      </div>
    </section>
  );
}

function fmt(v: number | null | undefined) {
  return v == null ? "—" : Number(v).toFixed(3);
}
