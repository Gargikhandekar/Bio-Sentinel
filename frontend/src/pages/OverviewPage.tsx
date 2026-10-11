import { Link } from "react-router-dom";
import { AdaptiveCore } from "../components/AdaptiveCore";
import { Disclaimer } from "../components/Primitives";

export function OverviewPage() {
  return (
    <section className="px-6 py-10 md:px-10">
      <p className="mono text-[11px] tracking-[0.35em] text-aqua">ADAPTIVE COMPUTATIONAL INTELLIGENCE</p>
      <h1 className="mt-3 text-4xl font-medium tracking-tight md:text-6xl">BIO-SENTINEL</h1>
      <p className="mt-4 max-w-2xl text-lg text-mute">
        Chronic disease risk, continuously re-evaluated. The system combines heterogeneous health features,
        adaptive learning, dynamic model selection and uncertainty-aware prediction.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          to="/predict"
          className="rounded-md bg-aqua px-5 py-2 text-sm font-medium text-ink hover:opacity-90"
        >
          Run Prediction
        </Link>
        <Link to="/research" className="rounded-md border border-white/15 px-5 py-2 text-sm text-text">
          Open research console
        </Link>
      </div>
      <div className="mt-12 rounded-xl border border-white/10 bg-panel/80 p-6">
        <p className="mono text-[11px] tracking-[0.25em] text-mute">ADAPTIVE INTELLIGENCE CORE</p>
        <AdaptiveCore nodes={[]} />
      </div>
      <div className="mt-8 max-w-3xl">
        <Disclaimer />
      </div>
    </section>
  );
}
