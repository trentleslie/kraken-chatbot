import { useEffect, useRef } from "react";
import { ClerkProvider, useClerk, useUser } from "@clerk/react";
import { Switch, Route, useLocation, Redirect } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import ChatPage from "@/pages/chat";
import SharedConversation from "@/pages/SharedConversation";
import LoginPage from "@/pages/Login";
import SignUpPage from "@/pages/SignUp";
import ProtectedRoute from "@/components/ProtectedRoute";

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
const clerkProxyUrl = import.meta.env.VITE_CLERK_PROXY_URL;

/**
 * Invalidate React Query cache when Clerk user changes (sign in/out).
 * Prevents stale data from leaking between sessions.
 */
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
      publishableKey={clerkPubKey}
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

function AppWithoutClerk() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Switch>
          <Route path="/" component={ChatPage} />
          <Route path="/:conversationId" component={SharedConversation} />
        </Switch>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

function App() {
  // When Clerk publishable key is not set, render without auth (local dev mode)
  if (!clerkPubKey) {
    return <AppWithoutClerk />;
  }
  return <AppWithClerk />;
}

export default App;
