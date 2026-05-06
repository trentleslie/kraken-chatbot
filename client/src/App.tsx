import { Show, SignInButton, UserButton } from "@clerk/react";
import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import ChatPage from "@/pages/chat";
import SharedConversation from "@/pages/SharedConversation";
import ProtectedRoute from "@/components/ProtectedRoute";

const clerkEnabled = !!import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

function AuthHeader() {
  return (
    <div className="fixed top-4 right-4 z-50">
      <Show when="signed-out">
        <SignInButton />
      </Show>
      <Show when="signed-in">
        <UserButton />
      </Show>
    </div>
  );
}

function Router() {
  if (!clerkEnabled) {
    // No auth — render routes directly (local dev mode)
    return (
      <Switch>
        <Route path="/" component={ChatPage} />
        <Route path="/:conversationId" component={SharedConversation} />
      </Switch>
    );
  }

  return (
    <Switch>
      <Route path="/">
        {() => <ProtectedRoute component={ChatPage} />}
      </Route>
      <Route path="/:conversationId">
        {() => <ProtectedRoute component={SharedConversation} />}
      </Route>
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        {clerkEnabled && <AuthHeader />}
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
