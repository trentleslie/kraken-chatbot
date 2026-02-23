import { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FeedbackButtonsProps {
  turnId?: string;
  conversationId: string | null;
  traceId?: string;
  disabled?: boolean;
}

type FeedbackType = "positive" | "negative" | null;

export function FeedbackButtons({
  turnId,
  conversationId,
  traceId,
  disabled = false,
}: FeedbackButtonsProps) {
  const [submittedFeedback, setSubmittedFeedback] = useState<FeedbackType>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedback = async (feedbackType: "positive" | "negative") => {
    if (!turnId || !conversationId || disabled) return;

    setIsSubmitting(true);

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          turn_id: turnId,
          conversation_id: conversationId,
          feedback_type: feedbackType,
          trace_id: traceId,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit feedback");
      }

      setSubmittedFeedback(feedbackType);
    } catch (error) {
      console.error("Error submitting feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Don't show buttons if no turnId or in demo mode (no conversationId)
  if (!turnId || !conversationId || disabled) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 mt-2">
      <Button
        variant={submittedFeedback === "positive" ? "default" : "outline"}
        size="sm"
        onClick={() => handleFeedback("positive")}
        disabled={isSubmitting || submittedFeedback !== null}
        className="h-7 px-2"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant={submittedFeedback === "negative" ? "default" : "outline"}
        size="sm"
        onClick={() => handleFeedback("negative")}
        disabled={isSubmitting || submittedFeedback !== null}
        className="h-7 px-2"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
      {submittedFeedback && (
        <span className="text-xs text-muted-foreground ml-1">
          Thanks for your feedback!
        </span>
      )}
    </div>
  );
}
