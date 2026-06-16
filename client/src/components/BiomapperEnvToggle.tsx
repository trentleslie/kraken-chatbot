import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FlaskConical } from "lucide-react";
import type { BiomapperEnv } from "@/types/messages";

interface BiomapperEnvToggleProps {
  env: BiomapperEnv;
  onEnvChange: (env: BiomapperEnv) => void;
  disabled?: boolean;
}

interface EnvButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
  disabled?: boolean;
}

function EnvButton({ label, isActive, onClick, disabled }: EnvButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
        isActive
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      {label}
    </button>
  );
}

/**
 * Prod/Dev toggle for the biomapper2 entity-resolution API used by the discovery pipeline.
 * Mirrors biomapper-ui's environment toggle: the selected env is sent as `biomapper_env` on the
 * WebSocket user_message and routes the backend pre-resolver to the prod or dev biomapper2 instance.
 */
export function BiomapperEnvToggle({ env, onEnvChange, disabled }: BiomapperEnvToggleProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1.5" data-testid="biomapper-env-toggle">
            <FlaskConical className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Biomapper</span>
            <div className="flex items-center gap-1 p-0.5 bg-muted rounded-md">
              <EnvButton
                label="Prod"
                isActive={env === "production"}
                onClick={() => onEnvChange("production")}
                disabled={disabled}
              />
              <EnvButton
                label="Dev"
                isActive={env === "dev"}
                onClick={() => onEnvChange("dev")}
                disabled={disabled}
              />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs">
          <p>
            Which biomapper2 API the discovery pipeline uses for entity resolution.{" "}
            <strong>Dev</strong> includes the HGNC human-gene fix (species-correct CURIEs);{" "}
            <strong>Prod</strong> is the stable endpoint. Only affects Discovery Pipeline runs.
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
