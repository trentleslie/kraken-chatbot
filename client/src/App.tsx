import { useEffect, useRef } from "react";
import { ClerkProvider, SignIn, SignUp, useClerk, useUser } from "@clerk/react";
import { Switch, Route, useLocation, Redirect } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import ChatPage from "@/pages/chat";
import SharedConversation from "@/pages/SharedConversation";
import AccessDeniedPage from "@/pages/AccessDenied";

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as string | undefined;
const clerkProxyUrl = import.meta.env.VITE_CLERK_PROXY_URL as string | undefined;

function LoginPage() {
  return (
    <div style={{ display: "flex", justifyContent: "center", marginTop: "2rem" }}>
      <SignIn routing="path" path="/login" signUpUrl="/sign-up" />
    </div>
  );
}

function SignUpPage() {
  return (
    <div style={{ display: "flex", justifyContent: "center", marginTop: "2rem" }}>
      <SignUp routing="path" path="/sign-up" signInUrl="/login" />
    </div>
  );
}

function ClerkQueryClientCacheInvalidator() {
  const { addListener } = useClerk();
  const qc = useQueryClient();
  const prevUserIdRef = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    const unsubscribe = addListener(({ user }) => {
      const userId = user?.id ?? null;
      if (prevUserIdRef.current !== undefined && prevUserIdRef.current !== userId) {
        qc.clear();
      }
      prevUserIdRef.current = userId;
    });
    return unsubscribe;
  }, [addListener, qc]);

  return null;
}

// Configurable allow-policy for UX gating (backend is authoritative source of truth).
const rawFrontendDomains = (import.meta.env.VITE_ALLOWED_EMAIL_DOMAINS as string | undefined) || "";
const ALLOWED_UX_DOMAINS = rawFrontendDomains.split(",").map((d: string) => d.trim().toLowerCase()).filter(Boolean);

const rawFrontendEmails = (import.meta.env.VITE_ALLOWED_EMAILS as string | undefined) || "";
const ALLOWED_UX_EMAILS = new Set(rawFrontendEmails.split(",").map((e: string) => e.trim().toLowerCase()).filter(Boolean));

function ClerkProtectedRoute({ component: Component }: { component: React.ComponentType }) {
  const { user, isLoaded, isSignedIn } = useUser();

  if (!isLoaded) return null;
  if (!isSignedIn) return <Redirect to="/login" />;

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

// When Clerk isn't configured (local dev / no publishable key), no ClerkProvider
// is mounted, so calling useUser() would throw. In that case bypass auth and
// render the route directly. `clerkPubKey` is a build-time constant, so this
// branch is stable across renders (no conditional-hooks violation), and prod —
// which always sets the key — is unaffected.
function ProtectedRoute({ component: Component }: { component: React.ComponentType }) {
  if (!clerkPubKey) return <Component />;
  return <ClerkProtectedRoute component={Component} />;
}

function Router() {
  return (
    <Switch>
      <Route path="/">
        {() => <ProtectedRoute component={ChatPage} />}
      </Route>

      <Route path="/login/*?" component={LoginPage} />

      <Route path="/sign-in/*?">
        {() => <Redirect to="/login" />}
      </Route>

      <Route path="/sign-up/*?" component={SignUpPage} />

      <Route path="/:conversationId">
        {() => <ProtectedRoute component={SharedConversation} />}
      </Route>
    </Switch>
  );
}

function AppWithClerk() {
  const [, setLocation] = useLocation();

  return (
    <ClerkProvider
      publishableKey={clerkPubKey!}
      proxyUrl={clerkProxyUrl}
      routerPush={(to) => setLocation(to)}
      routerReplace={(to) => setLocation(to, { replace: true })}
    >
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <ClerkQueryClientCacheInvalidator />
          <Router />
          <Toaster />
        </TooltipProvider>
      </QueryClientProvider>
    </ClerkProvider>
  );
}

function AppNoAuth() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router />
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

function App() {
  if (!clerkPubKey) return <AppNoAuth />;
  return <AppWithClerk />;
}

export default App;
