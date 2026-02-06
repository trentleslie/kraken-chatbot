interface MessageBubbleProps {
  content: string;
}

export function MessageBubble({ content }: MessageBubbleProps) {
  return (
    <div className="flex justify-end" data-testid="message-user">
      <div className="max-w-[75%] rounded-md bg-primary px-4 py-2.5">
        <p className="text-sm text-primary-foreground whitespace-pre-wrap leading-relaxed">
          {content}
        </p>
      </div>
    </div>
  );
}
