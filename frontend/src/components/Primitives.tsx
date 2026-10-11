export function Disclaimer() {
  return (
    <aside className="rounded-md border border-warn/30 bg-warn/5 px-4 py-3 text-xs text-mute">
      <p className="mono text-[10px] tracking-[0.2em] text-warn">RESEARCH PROTOTYPE</p>
      <p className="mt-1">
        This system estimates model-based disease risk from public research datasets. It is not a medical
        diagnostic device and must not be used for clinical decision-making.
      </p>
    </aside>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-dashed border-white/15 bg-panel px-6 py-10 text-center">
      <p className="mono text-sm text-aqua">{message}</p>
      <p className="mt-2 text-sm text-mute">Run the experiment pipeline to populate this view with measured values.</p>
    </div>
  );
}

export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-panel2 px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.16em] text-mute">{label}</p>
      <p className="mono mt-1 text-xl text-text">{value}</p>
    </div>
  );
}
