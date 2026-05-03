import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves at https://antonio-tresol.github.io/gemma3-refusal-axis/
// so all asset URLs need this prefix in production. In dev (vite serve) the base
// stays at "/" so paths work locally.
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/gemma3-refusal-axis/" : "/",
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  server: {
    port: 4002,
    host: "127.0.0.1",
  },
}));
