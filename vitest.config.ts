import { defineConfig } from "vitest/config";
import path from "path";

// Separate from vite.config.ts (which uses top-level await import() + root: client/ and is
// tuned for the app build). This config only powers the pure-logic unit tests (analyteParse).
export default defineConfig({
  test: {
    environment: "node",
    include: ["client/src/**/*.test.ts"],
  },
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
    },
  },
});
