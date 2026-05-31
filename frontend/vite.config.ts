import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server runs on :5173 — already allow-listed in the backend CORS config.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
