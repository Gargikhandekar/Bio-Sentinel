import { motion } from "framer-motion";

type Node = { name: string; probability?: number; weight?: number; competence?: number };

export function AdaptiveCore({ nodes }: { nodes: Node[] }) {
  const items = nodes.length ? nodes : [
    { name: "Online Learner" },
    { name: "Adaptive Tree" },
    { name: "Tabular Model" },
    { name: "Reference" },
  ];
  return (
    <div className="relative mx-auto h-72 w-full max-w-xl" aria-label="Adaptive intelligence core">
      {items.map((node, i) => {
        const angle = (i / items.length) * Math.PI * 2 - Math.PI / 2;
        const x = 50 + Math.cos(angle) * 38;
        const y = 50 + Math.sin(angle) * 36;
        const width = 1 + (node.weight ?? 0.25) * 8;
        return (
          <div key={node.name}>
            <svg className="absolute inset-0 h-full w-full" aria-hidden>
              <line
                x1="50%"
                y1="50%"
                x2={`${x}%`}
                y2={`${y}%`}
                stroke="#55E6D2"
                strokeOpacity="0.35"
                strokeWidth={width}
              />
            </svg>
            <div
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-md border border-white/10 bg-panel2 px-2 py-1 text-center"
              style={{ left: `${x}%`, top: `${y}%` }}
              title={`${node.name} p=${node.probability ?? "—"} w=${node.weight ?? "—"}`}
            >
              <p className="mono text-[10px] text-mute">{node.name}</p>
              {node.probability != null && (
                <p className="mono text-xs text-aqua">{(node.probability * 100).toFixed(1)}%</p>
              )}
            </div>
          </div>
        );
      })}
      <motion.div
        className="absolute left-1/2 top-1/2 grid h-20 w-20 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-aqua/40 bg-ink"
        animate={{ boxShadow: ["0 0 0 0 rgba(85,230,210,0.0)", "0 0 24px 2px rgba(85,230,210,0.25)"] }}
        transition={{ duration: 2.4, repeat: Infinity, repeatType: "reverse" }}
      >
        <span className="mono text-[10px] text-aqua">ACWE</span>
      </motion.div>
    </div>
  );
}
