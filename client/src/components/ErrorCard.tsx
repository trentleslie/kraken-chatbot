import { AlertCircle } from "lucide-react";
import { Card } from "@/components/ui/card";

interface ErrorCardProps {
  message: string;
}

export function ErrorCard({ message }: ErrorCardProps) {
  return (
    <div className="flex justify-start" data-testid="message-error">
      <Card className="max-w-[85%] bg-destructive/10 border-destructive/20">
        <div className="flex items-start gap-2.5 p-3">
          <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{message}</p>
        </div>
      </Card>
    </div>
  );
}
