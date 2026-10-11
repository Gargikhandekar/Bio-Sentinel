import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { EmptyState } from "../components/Primitives";
import { api } from "../services/api";

export function DriftPage() {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    api.drift().then(setData).catch(() => setData({ status: "Experiment not yet executed" }));
  }, []);
  if (!data || data.status === "Experiment not yet executed") {
    return <section className="p-8"><EmptyState message="Experiment not yet executed" /></section>;
  }
  const rolling = data.rolling || [];
  return (
    <section className="px-4 py-8 lg:px-10">
      <h1 className="text-2xl">DRIFT MONITOR</h1>
      <p className="mt-2 max-w-3xl text-sm text-mute">{data.note}</p>
      <div className="mt-6 rounded-xl border border-white/10 bg-panel p-3">
        <ReactECharts
          style={{ height: 320 }}
          option={{
            backgroundColor: "transparent",
            tooltip: { trigger: "axis" },
            xAxis: { type: "value", name: "t", axisLabel: { color: "#9AA6B2" } },
            yAxis: { type: "value", min: 0.5, max: 1, axisLabel: { color: "#9AA6B2" } },
            series: [
              {
                type: "line",
                showSymbol: false,
                data: rolling.map((r: any) => [r.t, r.rolling_accuracy]),
                lineStyle: { color: "#55E6D2" },
                markLine: data.injected_at
                  ? { data: [{ xAxis: data.injected_at, label: { formatter: "DRIFT INJECTED", color: "#FF6B7A" } }], lineStyle: { color: "#FF6B7A" } }
                  : undefined,
              },
            ],
          }}
        />
      </div>
      <div className="mt-6 grid gap-3 md:grid-cols-2">
        {(data.drift_events || []).map((event: any, i: number) => (
          <article key={i} className="rounded-xl border border-white/10 bg-panel2 p-4 text-sm">
            <p className="mono text-aqua">{event.detector}</p>
            <p>stream position {event.stream_position}</p>
            <p className="text-mute">{event.note}</p>
          </article>
        ))}
        {!data.drift_events?.length && <p className="text-mute">No detector trips were recorded on this run.</p>}
      </div>
    </section>
  );
}
