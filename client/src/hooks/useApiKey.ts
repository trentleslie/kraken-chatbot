import { useState, useCallback } from "react";

export type ValidateResult =
  | { valid: true }
  | { valid: false; reason: string };

export interface UseApiKeyReturn {
  /** Current in-memory key for this session (null = not set). */
  key: string | null;
  /** Store the key in memory (never persisted to any storage). */
  setKey: (key: string) => void;
  /** Clear the in-memory key. */
  clearKey: () => void;
  /**
   * POST /api/validate-key and return the result.
   * Does NOT set the key; caller decides what to do on success.
   */
  validate: (key: string) => Promise<ValidateResult>;
}

export function useApiKey(): UseApiKeyReturn {
  // Key lives ONLY in React state — no localStorage, no sessionStorage, no cookies.
  // A full page reload intentionally clears it (per-session design).
  const [key, setKeyState] = useState<string | null>(null);

  const setKey = useCallback((k: string) => {
    setKeyState(k);
  }, []);

  const clearKey = useCallback(() => {
    setKeyState(null);
  }, []);

  const validate = useCallback(async (k: string): Promise<ValidateResult> => {
    try {
      const res = await fetch("/api/validate-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: k }),
      });
      const json = (await res.json()) as { valid: boolean; reason?: string };
      if (json.valid) {
        return { valid: true };
      }
      return { valid: false, reason: json.reason ?? "Invalid key" };
    } catch {
      return { valid: false, reason: "Network error — could not reach validation endpoint" };
    }
  }, []);

  return { key, setKey, clearKey, validate };
}
