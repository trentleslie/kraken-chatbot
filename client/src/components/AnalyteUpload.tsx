import { useCallback, useRef, useState, type DragEvent, type KeyboardEvent } from "react";
import { Loader2, Upload, AlertCircle } from "lucide-react";
import { parseFile, type ParsedFile } from "@/lib/analyteParse";

interface AnalyteUploadProps {
  /** Called with the parsed file on a successful parse. */
  onParsed: (parsed: ParsedFile, fileName: string) => void;
  /** Idle-state primary prompt (defaults to the analyte-panel copy). */
  idlePrimary?: string;
  /** Idle-state secondary hint (defaults to the analyte-panel copy). */
  idleSecondary?: string;
  /** Accessible label for the dropzone (defaults to the analyte-panel copy). */
  ariaLabel?: string;
  /** data-testid for the dropzone / input (defaults to the analyte-panel ids). */
  testId?: string;
}

type UploadState =
  | { status: "idle" }
  | { status: "dragging" }
  | { status: "parsing" }
  | { status: "error"; message: string };

/**
 * Drag-and-drop (with click-to-browse fallback) target for a CSV/TSV analyte file (Unit 6).
 * Parsing is client-side (analyteParse); the raw file never reaches the backend. Pinned states:
 * idle / drag-over / parsing / error. Accessible: role="button", tabIndex, Enter/Space opens picker.
 */
export function AnalyteUpload({
  onParsed,
  idlePrimary = "Drop a CSV or TSV analyte file, or browse",
  idleSecondary = "Map the analyte, group, and type columns after upload",
  ariaLabel = "Upload analyte file: drop a CSV or TSV file, or press Enter to browse",
  testId = "analyte",
}: AnalyteUploadProps) {
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setState({ status: "parsing" });
      const result = await parseFile(file);
      if (!result.ok) {
        setState({ status: "error", message: result.error });
        return;
      }
      setState({ status: "idle" });
      onParsed(result.data, file.name);
    },
    [onParsed],
  );

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file) void handleFile(file);
      else setState({ status: "idle" });
    },
    [handleFile],
  );

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setState((s) => (s.status === "parsing" ? s : { status: "dragging" }));
  }, []);

  const onDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setState((s) => (s.status === "dragging" ? { status: "idle" } : s));
  }, []);

  const openPicker = useCallback(() => {
    if (state.status !== "parsing") inputRef.current?.click();
  }, [state.status]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openPicker();
      }
    },
    [openPicker],
  );

  const isDragging = state.status === "dragging";
  const isParsing = state.status === "parsing";
  const isError = state.status === "error";

  return (
    <div className="max-w-3xl mx-auto">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.tab"
        className="hidden"
        data-testid={`${testId}-file-input`}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          // Reset so re-selecting the same file re-fires change.
          e.target.value = "";
        }}
      />
      <div
        role="button"
        tabIndex={0}
        aria-label={ariaLabel}
        aria-disabled={isParsing}
        onClick={openPicker}
        onKeyDown={onKeyDown}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        data-testid={`${testId}-dropzone`}
        className={[
          "flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed",
          "px-4 py-6 text-sm cursor-pointer transition-colors focus:outline-none focus:ring-2 focus:ring-ring",
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/30",
          isError ? "border-destructive bg-destructive/5" : "",
          isParsing ? "cursor-wait opacity-80" : "",
        ].join(" ")}
      >
        {isParsing ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="text-muted-foreground">Parsing file…</span>
          </>
        ) : isError ? (
          <>
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-destructive text-center">{state.message}</span>
            <span className="text-xs text-muted-foreground">Click or drop to try another file</span>
          </>
        ) : isDragging ? (
          <>
            <Upload className="h-5 w-5 text-primary" />
            <span className="text-primary">Release to upload</span>
          </>
        ) : (
          <>
            <Upload className="h-5 w-5 text-muted-foreground" />
            <span className="text-muted-foreground">{idlePrimary}</span>
            <span className="text-xs text-muted-foreground">{idleSecondary}</span>
          </>
        )}
      </div>
    </div>
  );
}
