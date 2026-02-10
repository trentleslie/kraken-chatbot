import type { PipelineProgress as PipelineProgressType } from "@/types/messages";

interface PipelineProgressProps {
  progress: PipelineProgressType;
}

export function PipelineProgress({ progress }: PipelineProgressProps) {
  const percentage = (progress.nodesCompleted / progress.totalNodes) * 100;

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-muted/50 rounded-lg border border-border/50">
      <div className="animate-pulse text-lg">🔬</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-muted-foreground truncate">{progress.message}</p>
        <div className="h-1.5 bg-muted rounded-full mt-1.5 overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-500 ease-out rounded-full"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
      <span className="text-xs text-muted-foreground font-mono">
        {progress.nodesCompleted}/{progress.totalNodes}
      </span>
    </div>
  );
}
