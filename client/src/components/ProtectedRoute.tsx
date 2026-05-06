import { useUser, SignInButton } from "@clerk/react";
import AccessDeniedPage from "@/pages/AccessDenied";

// Configurable allow-policy for UX gating (backend is authoritative source of truth).
// Override via VITE_ALLOWED_EMAIL_DOMAINS or VITE_ALLOWED_EMAILS (comma-separated) env vars.
const rawDomains = (import.meta.env.VITE_ALLOWED_EMAIL_DOMAINS as string | undefined) || "";
const ALLOWED_UX_DOMAINS = rawDomains.split(",").map((d: string) => d.trim().toLowerCase()).filter(Boolean);

const rawEmails = (import.meta.env.VITE_ALLOWED_EMAILS as string | undefined) || "";
const ALLOWED_UX_EMAILS = new Set(rawEmails.split(",").map((e: string) => e.trim().toLowerCase()).filter(Boolean));

export default function ProtectedRoute({ component: Component }: { component: React.ComponentType }) {
  const { user, isLoaded, isSignedIn } = useUser();

  if (!isLoaded) return null;

  if (!isSignedIn) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6 text-center">
        <h1 className="text-2xl font-bold tracking-tight mb-4">Sign in to continue</h1>
        <p className="text-muted-foreground mb-6">Authentication is required to access this application.</p>
        <SignInButton />
      </div>
    );
  }

  // If no domain restrictions configured, allow all authenticated users
  if (ALLOWED_UX_DOMAINS.length === 0 && ALLOWED_UX_EMAILS.size === 0) {
    return <Component />;
  }

  const email = (user.primaryEmailAddress?.emailAddress || "").toLowerCase();
  const emailDomain = email.split("@")[1] || "";
  const isAllowed = ALLOWED_UX_EMAILS.has(email) || ALLOWED_UX_DOMAINS.includes(emailDomain);

  if (!isAllowed) {
    return <AccessDeniedPage />;
  }

  return <Component />;
}
