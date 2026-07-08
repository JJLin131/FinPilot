import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        carbon: "#03080d",
        ink: "#061421",
        cyanflow: "#35d7d0",
        risk: "#f5b451",
      },
    },
  },
  plugins: [],
} satisfies Config;
