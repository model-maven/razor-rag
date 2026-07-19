import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development, requests to /api are forwarded to the FastAPI server,
// so the frontend and backend behave like one app.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
