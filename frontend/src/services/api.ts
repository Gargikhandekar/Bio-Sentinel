export type PredictResponse = {
  prediction: number;
  raw_probability: number;
  calibrated_probability: number;
  uncertainty: string;
  prediction_set: number[];
  model_predictions: Record<string, number>;
  model_weights: Record<string, number>;
  explanation: { disclaimer?: string; local?: Array<{ feature: string; contribution: number; direction: string; magnitude: number }> };
  disclaimer: string;
};

const API = "";

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  health: () => fetch(`${API}/api/health`).then((r) => parse<{ status: string; services: Record<string, string> }>(r)),
  datasets: () => fetch(`${API}/api/datasets`).then((r) => parse(r)),
  models: () => fetch(`${API}/api/models`).then((r) => parse(r)),
  model: (id: string) => fetch(`${API}/api/models/${encodeURIComponent(id)}`).then((r) => parse(r)),
  experiments: () => fetch(`${API}/api/experiments`).then((r) => parse(r)),
  drift: () => fetch(`${API}/api/drift`).then((r) => parse(r)),
  researchMetrics: () => fetch(`${API}/api/research/metrics`).then((r) => parse(r)),
  ablation: () => fetch(`${API}/api/research/ablation`).then((r) => parse(r)),
  calibration: () => fetch(`${API}/api/research/calibration`).then((r) => parse(r)),
  predict: (features: Record<string, number | null>, label = "diabetes") =>
    fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, label }),
    }).then((r) => parse<PredictResponse>(r)),
  explain: (features: Record<string, number | null>) =>
    fetch(`${API}/api/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, label: "diabetes" }),
    }).then((r) => parse(r)),
};
