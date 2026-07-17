import { useState } from "react";
import { AlertCircle } from "lucide-react";
import { Header } from "@/components/Header";
import { ChatArea } from "@/components/ChatArea";
import { ChatInput } from "@/components/ChatInput";
import { ModeToggle } from "@/components/ModeToggle";
import { BiomapperEnvToggle } from "@/components/BiomapperEnvToggle";
import { PipelineProgress } from "@/components/PipelineProgress";
import { AnalyteUpload } from "@/components/AnalyteUpload";
import { ColumnMappingPanel } from "@/components/ColumnMappingPanel";
import { DirectionMappingPanel } from "@/components/DirectionMappingPanel";
import { AnalyteReviewSummary } from "@/components/AnalyteReviewSummary";
import { ApiKeyGate } from "@/components/ApiKeyGate";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useApiKey } from "@/hooks/useApiKey";
import { Badge } from "@/components/ui/badge";
import { distinctGroups, type ParsedFile } from "@/lib/analyteParse";
import type { ErrorMessage } from "@/types/messages";

// Upload flow stages within the collapsible composer slot (pipeline mode only).
// `directions` maps the (separately-exported) ME-trait direction table after the analyte panel is
// staged, so per-module directions reach the backend and the sign-inversion metric is computable.
type UploadStage =
  | { step: "idle" }
  | { step: "mapping"; parsed: ParsedFile; fileName: string }
  | { step: "directions"; parsed: ParsedFile; fileName: string };

export default function ChatPage() {
  // BYOK: key lives only in React state (never persisted).
  const apiKey = useApiKey();

  const {
    messages,
    connectionStatus,
    isAgentResponding,
    sessionStats,
    conversationId,
    agentMode,
    setAgentMode,
    biomapperEnv,
    setBiomapperEnv,
    pipelineProgress,
    structuredAnalytes,
    setStructuredAnalytes,
    selectedGroups,
    setSelectedGroups,
    moduleDirections,
    setModuleDirections,
    sendMessage,
    clearMessages,
    needsKey,
    keySource,
  } = useWebSocket({ apiKey: apiKey.key });

  const [uploadStage, setUploadStage] = useState<UploadStage>({ step: "idle" });
  const [queryEmpty, setQueryEmpty] = useState(true);

  const isConnected = connectionStatus === "connected" || connectionStatus === "demo";
  const isPipeline = agentMode === "pipeline";
  const hasPanel = structuredAnalytes.length > 0;

  // Check for AUTH_ERROR in messages
  const hasAuthError = messages.some(
    (m) => m.type === "error" && (m as ErrorMessage).code === "AUTH_ERROR"
  );

  // BYOK: composer is additionally blocked when the server needs a key and none is set.
  const keyGateBlocking = needsKey && apiKey.key === null;

  const resetUpload = () => {
    setUploadStage({ step: "idle" });
    setStructuredAnalytes([]);
    setSelectedGroups([]);
    setModuleDirections([]);
  };

  const handleSelectStarter = (query: string) => {
    if (isConnected && !isAgentResponding) {
      sendMessage(query);
    }
  };

  const handleSend = (query: string) => {
    sendMessage(query);
    // sendMessage clears the hook's panel/selection after a successful send (one-shot);
    // collapse the composer slot back to the drop zone.
    setUploadStage({ step: "idle" });
  };

  // The review summary shows once analytes are staged and mapping is done.
  const showReview = hasPanel && uploadStage.step === "idle";

  return (
    <div className="flex flex-col h-screen bg-background">
      {hasAuthError && (
        <div className="bg-destructive text-destructive-foreground px-4 py-3 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm font-medium">
            Server authentication has expired. Please contact the administrator to re-authenticate.
          </span>
        </div>
      )}
      <Header
        connectionStatus={connectionStatus}
        sessionStats={sessionStats}
        onClearChat={clearMessages}
        hasMessages={messages.length > 0}
        conversationId={conversationId}
      />

      {/* Mode Toggle - below header, above messages */}
      <div className="px-4 py-2 border-b flex items-center justify-center gap-4">
        <ModeToggle
          mode={agentMode}
          onModeChange={setAgentMode}
          disabled={isAgentResponding}
        />
        {/* Biomapper prod/dev API toggle — only relevant to the discovery pipeline's resolver. */}
        {isPipeline && (
          <BiomapperEnvToggle
            env={biomapperEnv}
            onEnvChange={setBiomapperEnv}
            disabled={isAgentResponding}
          />
        )}
        {/* BYOK: key-source badge — shown when the server has confirmed which key is in use. */}
        {keySource !== null && (
          <Badge variant={keySource === "byok" ? "default" : "secondary"} className="text-xs">
            {keySource === "byok" ? "Using your key" : "Using server key"}
          </Badge>
        )}
      </div>

      <ChatArea
        messages={messages}
        isAgentResponding={isAgentResponding}
        isConnected={isConnected}
        conversationId={conversationId}
        onSelectStarter={handleSelectStarter}
      />

      {/* Pipeline Progress - above input when active */}
      {pipelineProgress && (
        <div className="px-4 pb-2">
          <PipelineProgress progress={pipelineProgress} />
        </div>
      )}

      {/* Analyte upload slot — pipeline mode only, between the mode row and the textarea.
          Collapses to zero height when no upload is in progress. Sequence: drop → mapping → review. */}
      {isPipeline && !hasAuthError && (
        <div className="px-4 pb-2">
          {uploadStage.step === "idle" && !showReview && (
            <AnalyteUpload
              onParsed={(parsed, fileName) => setUploadStage({ step: "mapping", parsed, fileName })}
            />
          )}
          {uploadStage.step === "mapping" && (
            <ColumnMappingPanel
              parsed={uploadStage.parsed}
              fileName={uploadStage.fileName}
              onCancel={() => setUploadStage({ step: "idle" })}
              onConfirm={(analytes) => {
                setStructuredAnalytes(analytes);
                // Default the group selection to all groups present.
                setSelectedGroups(distinctGroups(analytes));
                setUploadStage({ step: "idle" });
              }}
            />
          )}
          {uploadStage.step === "directions" && (
            <DirectionMappingPanel
              parsed={uploadStage.parsed}
              fileName={uploadStage.fileName}
              onCancel={() => setUploadStage({ step: "idle" })}
              onConfirm={(directions) => {
                // Wire the parsed ME-trait table into the hook so `module_directions` actually
                // travels on the WS payload — without this the backend receives no directions and
                // the module-spine sign-inversion metric is uncomputable from the client path.
                setModuleDirections(directions);
                setUploadStage({ step: "idle" });
              }}
            />
          )}
          {showReview && (
            <div className="space-y-2">
              <AnalyteReviewSummary
                analytes={structuredAnalytes}
                selectedGroups={selectedGroups}
                onSelectedGroupsChange={setSelectedGroups}
                onRemove={resetUpload}
                queryEmpty={queryEmpty}
                directionCount={moduleDirections.length}
                onRemoveDirections={() => setModuleDirections([])}
              />
              {/* Optional second upload: the (separately-exported) per-module ME-trait direction
                  table. Attaching it lets the backend compute the sign-inversion metric. */}
              {moduleDirections.length === 0 && (
                <AnalyteUpload
                  testId="direction"
                  ariaLabel="Upload module-direction table: drop a CSV or TSV file, or press Enter to browse"
                  idlePrimary="Optional: drop the ME-trait direction table (CSV/TSV), or browse"
                  idleSecondary="One row per module — map the module, correlation, and trait columns after upload"
                  onParsed={(parsed, fileName) =>
                    setUploadStage({ step: "directions", parsed, fileName })
                  }
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* BYOK: key gate — blocks composer when server sends NEEDS_KEY and no key is set,
          or shows a "clear key" control when a key is already loaded for this session. */}
      {!hasAuthError && (
        <div className="px-4 pb-2">
          <ApiKeyGate apiKey={apiKey} needsKey={needsKey} />
        </div>
      )}

      <ChatInput
        onSend={handleSend}
        disabled={isAgentResponding || hasAuthError || keyGateBlocking}
        isConnected={isConnected}
        hasPanel={isPipeline && hasPanel}
        onQueryEmptyChange={setQueryEmpty}
      />
    </div>
  );
}
