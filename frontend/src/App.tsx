import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layouts/AppShell";
import { OverviewPage } from "./pages/OverviewPage";
import { PredictPage } from "./pages/PredictPage";
import { ModelLabPage } from "./pages/ModelLabPage";
import { DriftPage } from "./pages/DriftPage";
import { ExplainPage } from "./pages/ExplainPage";
import { ResearchPage } from "./pages/ResearchPage";
import { AblationPage } from "./pages/AblationPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { SystemPage } from "./pages/SystemPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/models" element={<ModelLabPage />} />
        <Route path="/drift" element={<DriftPage />} />
        <Route path="/explain" element={<ExplainPage />} />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="/research/ablation" element={<AblationPage />} />
        <Route path="/research/calibration" element={<CalibrationPage />} />
        <Route path="/system" element={<SystemPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
