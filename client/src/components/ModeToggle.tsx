import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import type { AgentMode } from "@/types/messages";

interface ModeToggleProps {
  mode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
  disabled?: boolean;
}

interface ModeButtonProps {
  label: string;
  tooltip: string;
  isActive: boolean;
  onClick: () => void;
  disabled?: boolean;
}

function ModeButton({ label, tooltip, isActive, onClick, disabled }: ModeButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={onClick}
          disabled={disabled}
          className={cn(
            "flex items-center gap-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
            isActive
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          {label}
          <Info className="h-3.5 w-3.5 opacity-50" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        <p>{tooltip}</p>
      </TooltipContent>
    </Tooltip>
  );
}

export function ModeToggle({ mode, onModeChange, disabled }: ModeToggleProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
        <ModeButton
          label="Classic"
          tooltip="Single AI agent with full knowledge graph access. Best for quick lookups and focused questions about specific entities."
          isActive={mode === "classic"}
          onClick={() => onModeChange("classic")}
          disabled={disabled}
        />
        <ModeButton
          label="Discovery Pipeline"
          tooltip="Multi-stage analysis pipeline with parallel entity resolution, novelty scoring, and hypothesis generation. Best for analyzing panels of metabolites, proteins, or genes."
          isActive={mode === "pipeline"}
          onClick={() => onModeChange("pipeline")}
          disabled={disabled}
        />
      </div>
    </TooltipProvider>
  );
}
