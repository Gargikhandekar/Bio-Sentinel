/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#080B0F",
        panel: "#0E1318",
        panel2: "#12181F",
        line: "rgba(255,255,255,0.08)",
        aqua: "#55E6D2",
        signal: "#7AA7FF",
        warn: "#F5B94C",
        crit: "#FF6B7A",
        text: "#F3F6F8",
        mute: "#9AA6B2",
      },
      fontFamily: {
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
