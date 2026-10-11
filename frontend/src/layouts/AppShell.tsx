import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  Activity,
  Binary,
  Cpu,
  FlaskConical,
  Radar,
  Search,
  Settings2,
} from "lucide-react";
import { api } from "../services/api";

const LINKS = [
  { to: "/", label: "Overview", icon: Activity },
  { to: "/predict", label: "Predict", icon: Binary },
  { to: "/models", label: "Model Lab", icon: Cpu },
  { to: "/drift", label: "Drift Monitor", icon: Radar },
  { to: "/explain", label: "Explainability", icon: Search },
  { to: "/research", label: "Research", icon: FlaskConical },
  { to: "/system", label: "System", icon: Settings2 },
];

export function AppShell() {
  const [health, setHealth] = useState<"operational" | "degraded" | "offline">("offline");
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    api
      .health()
      .then((h) => setHealth(h.status === "operational" ? "operational" : "degraded"))
      .catch(() => setHealth("offline"));
  }, [location.pathname]);

  return (
    <div className="min-h-screen grid-bg text-text">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:p-2">
        Skip to content
      </a>
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-60 border-r border-white/10 bg-panel/95 px-4 py-6 md:block">
        <div className="mb-8 px-2">
          <p className="mono text-[11px] tracking-[0.28em] text-aqua">BIO-SENTINEL</p>
          <p className="mt-1 text-xs text-mute">Chronic disease risk, re-evaluated.</p>
        </div>
        <nav aria-label="Primary" className="space-y-1">
          {LINKS.map((link) => {
            const Icon = link.icon;
            return (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                    isActive ? "bg-white/5 text-aqua" : "text-mute hover:text-text"
                  }`
                }
              >
                <Icon size={16} aria-hidden />
                {link.label}
              </NavLink>
            );
          })}
        </nav>
      </aside>
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-white/10 bg-ink/80 px-4 py-3 backdrop-blur md:ml-60">
        <p className="mono text-[11px] uppercase tracking-[0.2em] text-mute md:hidden">BIO-SENTINEL</p>
        <p className="hidden text-sm text-mute md:block">Adaptive computational intelligence laboratory</p>
        <button
          className="mono flex items-center gap-2 text-xs"
          onClick={() => navigate("/system")}
          aria-label="Open system status"
        >
          <span
            className={`h-2 w-2 rounded-full ${
              health === "operational" ? "bg-aqua" : health === "degraded" ? "bg-warn" : "bg-crit"
            }`}
          />
          {health === "operational" ? "SYSTEM OPERATIONAL" : health === "degraded" ? "SYSTEM DEGRADED" : "SYSTEM OFFLINE"}
        </button>
      </header>
      <div className="flex gap-2 overflow-x-auto border-b border-white/10 px-3 py-2 md:hidden">
        {LINKS.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `whitespace-nowrap rounded-full px-3 py-1 text-xs ${isActive ? "bg-white/10 text-aqua" : "text-mute"}`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </div>
      <main id="main" className="md:ml-60">
        <Outlet />
      </main>
    </div>
  );
}
