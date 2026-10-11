import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AdaptiveCore } from "../components/AdaptiveCore";
import { Disclaimer } from "../components/Primitives";
import { api, PredictResponse } from "../services/api";

const GROUPS = [
  {
    title: "DEMOGRAPHICS",
    fields: [
      { key: "age_group", label: "Age group", hint: "1 = 18–24 … 13 = 80+", min: 1, max: 13, step: 1, default: 7 },
      { key: "sex", label: "Sex", hint: "1 male, 2 female", min: 1, max: 2, step: 1, default: 1 },
      { key: "race_group", label: "Race group", hint: "BRFSS _RACEGR3", min: 1, max: 8, step: 1, default: 1 },
      { key: "education", label: "Education", hint: "1 < HS … 4 college", min: 1, max: 4, step: 1, default: 3 },
      { key: "income", label: "Income", hint: "1 <15k … 7 ≥200k", min: 1, max: 7, step: 1, default: 5 },
    ],
  },
  {
    title: "METABOLIC",
    fields: [
      { key: "bmi", label: "BMI", hint: "kg/m²", min: 12, max: 60, step: 0.1, default: 29.4 },
      { key: "high_cholesterol", label: "High cholesterol", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 1 },
      { key: "chol_check", label: "Cholesterol check", hint: "survey code", min: 1, max: 8, step: 1, default: 2 },
    ],
  },
  {
    title: "CARDIOVASCULAR / STATUS",
    fields: [
      { key: "gen_health", label: "General health", hint: "1 excellent … 5 poor", min: 1, max: 5, step: 1, default: 3 },
      { key: "phys_unhealthy_days", label: "Poor physical days", hint: "past 30 days", min: 0, max: 30, step: 1, default: 2 },
      { key: "ment_unhealthy_days", label: "Poor mental days", hint: "past 30 days", min: 0, max: 30, step: 1, default: 2 },
      { key: "difficulty_walking", label: "Difficulty walking", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 2 },
    ],
  },
  {
    title: "LIFESTYLE",
    fields: [
      { key: "smoker", label: "Smoker status", hint: "1 everyday … 4 never", min: 1, max: 4, step: 1, default: 4 },
      { key: "phys_activity", label: "Physical activity", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 1 },
      { key: "fruits_daily", label: "Fruit daily", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 1 },
      { key: "veggies_daily", label: "Vegetables daily", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 1 },
      { key: "healthcare_coverage", label: "Healthcare coverage", hint: "1 yes, 2 no", min: 1, max: 2, step: 1, default: 1 },
      { key: "binge_drinking", label: "Binge drinking", hint: "BRFSS _RFBING5", min: 1, max: 2, step: 1, default: 1 },
    ],
  },
];

const STEPS = [
  "FEATURE VECTOR",
  "MODEL POOL",
  "COMPETENCE ESTIMATION",
  "DYNAMIC WEIGHTING",
  "ENSEMBLE",
  "CALIBRATION",
  "UNCERTAINTY",
  "EXPLANATION",
];

export function PredictPage() {
  const defaults = useMemo(() => {
    const obj: Record<string, number> = {};
    GROUPS.forEach((g) => g.fields.forEach((f) => (obj[f.key] = f.default)));
    return obj;
  }, []);
  const [values, setValues] = useState(defaults);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [step, setStep] = useState(0);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    const messages = [
      "Evaluating model pool...",
      "Estimating local competence...",
      "Calibrating probability...",
      "Computing uncertainty...",
      "Generating explanation...",
    ];
    try {
      for (let i = 0; i < messages.length; i++) {
        setLoading(messages[i]);
        setStep(Math.min(STEPS.length - 1, i + 2));
        await new Promise((r) => setTimeout(r, 280));
      }
      const res = await api.predict(values);
      setResult(res);
      setStep(STEPS.length - 1);
    } catch (err) {
      setError("Unable to generate prediction. The model service is temporarily unavailable. Please retry.");
    } finally {
      setLoading(null);
    }
  }

  const marker = result ? result.calibrated_probability : 0.2;

  return (
    <section className="grid gap-8 px-4 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:px-10">
      <form onSubmit={onSubmit} className="space-y-6">
        <h1 className="text-2xl">CURRENT INFERENCE</h1>
        {GROUPS.map((group) => (
          <fieldset key={group.title} className="rounded-xl border border-white/10 bg-panel p-4">
            <legend className="mono px-1 text-[11px] tracking-[0.2em] text-aqua">{group.title}</legend>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {group.fields.map((field) => (
                <label key={field.key} className="text-sm">
                  <span className="text-mute">{field.label}</span>
                  <input
                    className="mono mt-1 w-full rounded-md border border-white/10 bg-ink px-3 py-2"
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={values[field.key]}
                    onChange={(e) => setValues((v) => ({ ...v, [field.key]: Number(e.target.value) }))}
                    aria-describedby={`${field.key}-hint`}
                    required
                  />
                  <span id={`${field.key}-hint`} className="text-xs text-mute">
                    {field.hint}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
        ))}
        <button className="rounded-md bg-aqua px-5 py-2 text-sm font-medium text-ink" type="submit" disabled={!!loading}>
          {loading ?? "Run prediction"}
        </button>
        {error && <p className="text-sm text-crit">{error}</p>}
      </form>
      <div className="space-y-6">
        <Disclaimer />
        <div className="rounded-xl border border-white/10 bg-panel p-5">
          <p className="text-xs uppercase tracking-[0.2em] text-mute">Predicted risk field</p>
          <div className="relative mt-8 h-16">
            <div className="absolute inset-x-0 top-6 h-px bg-white/20" />
            <p className="absolute left-0 top-0 text-xs text-mute">LOW</p>
            <p className="absolute right-0 top-0 text-xs text-mute">ELEVATED</p>
            <motion.div
              className="absolute top-4 h-5 w-5 -translate-x-1/2 rounded-full border border-aqua bg-ink"
              animate={{ left: `${marker * 100}%` }}
              transition={{ duration: 0.8 }}
            />
            <p className="mono absolute top-10 text-aqua" style={{ left: `${marker * 100}%`, transform: "translateX(-50%)" }}>
              {result ? `${(result.calibrated_probability * 100).toFixed(1)}%` : "—"}
            </p>
          </div>
          <div className="mt-8 grid grid-cols-3 gap-3 text-sm">
            <div>
              <p className="text-mute">Raw</p>
              <p className="mono">{result ? `${(result.raw_probability * 100).toFixed(1)}%` : "—"}</p>
            </div>
            <div>
              <p className="text-mute">Calibrated</p>
              <p className="mono">{result ? `${(result.calibrated_probability * 100).toFixed(1)}%` : "—"}</p>
            </div>
            <div>
              <p className="text-mute">Uncertainty</p>
              <p className="mono uppercase">{result?.uncertainty ?? "—"}</p>
            </div>
          </div>
        </div>
        <div className="rounded-xl border border-white/10 bg-panel p-5">
          <p className="mono text-[11px] tracking-[0.2em] text-mute">MODEL CONSENSUS</p>
          <AdaptiveCore
            nodes={
              result
                ? Object.entries(result.model_predictions).map(([name, probability]) => ({
                    name,
                    probability,
                    weight: result.model_weights[name],
                  }))
                : []
            }
          />
          {result && (
            <ul className="mt-4 space-y-1 text-sm">
              {Object.entries(result.model_predictions).map(([name, p]) => (
                <li key={name} className="flex justify-between">
                  <span>{name}</span>
                  <span className="mono text-aqua">
                    {(p * 100).toFixed(1)}% · w {(result.model_weights[name] * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <ol className="rounded-xl border border-white/10 bg-panel p-5 text-sm">
          {STEPS.map((item, i) => (
            <li key={item} className={`py-1 ${i <= step ? "text-aqua" : "text-mute"}`}>
              {item}
              {i < STEPS.length - 1 && <span className="block text-mute">↓</span>}
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
