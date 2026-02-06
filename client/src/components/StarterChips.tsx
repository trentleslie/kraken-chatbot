import { Button } from "@/components/ui/button";

interface StarterChipsProps {
  onSelect: (query: string) => void;
  disabled?: boolean;
}

const STARTER_QUERIES = [
  {
    label: "What is the CHEBI ID for glucose?",
    category: "Entity Resolution",
  },
  {
    label: "What pathways does glucose participate in?",
    category: "Relationships",
  },
  {
    label: "Find drugs that treat type 2 diabetes",
    category: "Combined Search",
  },
  {
    label: "What metabolites are similar to cholesterol?",
    category: "Similarity",
  },
  {
    label: "What genes interact with NAD+?",
    category: "Interactions",
  },
];

export function StarterChips({ onSelect, disabled }: StarterChipsProps) {
  return (
    <div
      className="flex flex-wrap gap-2 justify-center max-w-lg"
      data-testid="starter-chips"
    >
      {STARTER_QUERIES.map(({ label }) => (
        <Button
          key={label}
          variant="outline"
          size="sm"
          onClick={() => onSelect(label)}
          disabled={disabled}
          className="text-xs h-auto py-2 px-3 font-normal text-left whitespace-normal"
          data-testid={`chip-${label.slice(0, 20).replace(/\s+/g, "-").toLowerCase()}`}
        >
          {label}
        </Button>
      ))}
    </div>
  );
}
