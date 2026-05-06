import { SignIn } from "@clerk/react";

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <SignIn routing="path" path="/login" signUpUrl="/sign-up" />
    </div>
  );
}
