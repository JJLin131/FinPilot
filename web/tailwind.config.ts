import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ['"Playfair Display"', "Georgia", "serif"],
      },
      colors: {
        obsidian: "#0f1011",
        abyss: "#090a0b",
        graphite: "#2e2e2e",
        steel: "#3f4041",
        ash: "#9f9fa0",
        cloud: "#f5f5f7",
        pure: "#ffffff",
        "cyan-signal": "#00b3dd",
        "iris-gleam": "#847dff",
        orchid: "#dd90d8",
        periwinkle: "#90b8f0",
        governance: "#d8c58e",
      },
      backgroundImage: {
        "fin-gradient":
          "linear-gradient(135deg, #090a0b 0%, #0f1724 34%, #102f3a 62%, #261d35 100%)",
      },
    },
  },
  plugins: [],
} satisfies Config;
