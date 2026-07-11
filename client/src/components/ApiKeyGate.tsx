import { useState } from "react";
import { KeyRound, X, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import type { UseApiKeyReturn } from "@/hooks/useApiKey";

interface ApiKeyGateProps {
  /** Provided by useApiKey(). */
  apiKey: UseApiKeyReturn;
  /**
   * When true the gate renders as a blocking form (composer is disabled).
   * When false (a key is already set, or no NEEDS_KEY yet) the gate renders
   * only a "clear key" control.
   */
  needsKey: boolean;
}

export function ApiKeyGate({ apiKey, needsKey }: ApiKeyGateProps) {
  const { key, setKey, clearKey, validate } = apiKey;

  const [draft, setDraft] = useState("");
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleValidate = async () => {
    if (!draft.trim()) return;
    setValidating(true);
    setError(null);
    const result = await validate(draft.trim());
    setValidating(false);
    if (result.valid) {
      setKey(draft.trim());
      setDraft("");
    } else {
      setError(result.reason);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") void handleValidate();
  };

  // A key is set — show only the "clear key" affordance.
  if (key !== null) {
    return (
      <div className="flex items-center gap-2 px-1 py-1">
        <span className="text-xs text-muted-foreground">Your API key is set for this session.</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={clearKey}
          className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
          title="Clear API key"
        >
          <X className="h-3.5 w-3.5 mr-1" />
          Clear key
        </Button>
      </div>
    );
  }

  // Server hasn't asked for a key yet — nothing to render.
  if (!needsKey) return null;

  // The server returned NEEDS_KEY and no key is set — show the blocking form.
  return (
    <Card className="border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30">
      <div className="flex flex-col gap-3 p-4">
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
          <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
            An Anthropic API key is required to use this chat
          </p>
        </div>
        <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
          Paste your key below. It is held only in memory for this session and never stored or sent
          anywhere other than the server for this conversation.
        </p>

        <div className="flex gap-2">
          <Input
            type="password"
            placeholder="sk-ant-..."
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setError(null);
            }}
            onKeyDown={handleKeyDown}
            disabled={validating}
            className="font-mono text-sm"
            autoComplete="off"
            autoFocus
          />
          <Button
            onClick={() => void handleValidate()}
            disabled={!draft.trim() || validating}
            size="default"
            className="flex-shrink-0"
          >
            {validating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-1" />
                Checking…
              </>
            ) : (
              "Validate"
            )}
          </Button>
        </div>

        {error && (
          <div className="flex items-center gap-1.5 text-xs text-destructive">
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </Card>
  );
}
