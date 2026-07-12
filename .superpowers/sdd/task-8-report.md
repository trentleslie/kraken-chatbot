# Task 8 Implementation Report — Frontend BYOK Key Entry, WS Wiring, Source Badge

## Files Created

### `client/src/hooks/useApiKey.ts` (new)
Exports `useApiKey()` returning `{ key, setKey, clearKey, validate }`. Key is held exclusively in React state — no `localStorage`, `sessionStorage`, or cookies. `validate(key)` POSTs to `/api/validate-key` and returns a typed `ValidateResult`. A page reload intentionally clears the key (per-session design).

### `client/src/components/ApiKeyGate.tsx` (new)
A context-sensitive gate component that:
- Renders **nothing** when `needsKey` is false and no key is set (server hasn't asked yet).
- Renders a **blocking form** (amber card with password input + Validate button) when `needsKey` is true and `key === null`. Validates on button click or Enter key. Shows an inline error on failure. Sets the key on success.
- Renders a **"clear key" affordance only** (muted text + X button) when a key is already set, regardless of `needsKey`.

Uses `Button`, `Input`, and `Card` from `components/ui/` to match existing styling conventions. Uses lucide-react icons consistent with the codebase.

## Files Modified

### `client/src/types/messages.ts`
- Added `KeySource = "byok" | "server"` type.
- Added `SetKeyRequest` type for the outgoing `{"type":"set_key","key":...}` frame.
- Added `{ type: "key_source"; source: KeySource }` to the `IncomingMessage` union.
- The existing `{ type: "error"; ...; code?: string }` variant already covers `NEEDS_KEY` — no change needed there.

### `client/src/hooks/useWebSocket.ts`
- Accepts optional `{ apiKey?: string | null }` options parameter (defaults to `null`).
- Stores `apiKey` in a `useRef` so `onopen` always reads the current value without a stale closure.
- Added `sendSetKey(ws, key)` helper.
- `onopen`: sends `{"type":"set_key","key":...}` **before** any other message.
- `handleIncomingMessage` switch: handles `"key_source"` (updates `keySource` state; clears `needsKey` optimistically when `source === "byok"`). Handles `"error"` with `code === "NEEDS_KEY"` (sets `needsKey = true`, surfaces the error in chat, stops the responding spinner).
- Added a `useEffect` watching `apiKey` prop: when it changes mid-session, sends an updated `set_key` frame and clears `needsKey` optimistically if a key was provided.
- Returns `needsKey: boolean` and `keySource: KeySource | null` to consumers.

### `client/src/pages/chat.tsx`
- Adds `useApiKey()` at the top of `ChatPage`.
- Passes `{ apiKey: apiKey.key }` to `useWebSocket`.
- Destructures `needsKey` and `keySource` from the hook return.
- Computes `keyGateBlocking = needsKey && apiKey.key === null`.
- Renders `<ApiKeyGate apiKey={apiKey} needsKey={needsKey} />` above `ChatInput` (hidden behind `!hasAuthError` to avoid conflicting UI states).
- Renders a `<Badge>` ("Using your key" / "Using server key") in the mode toggle row whenever `keySource !== null`.
- Extends `ChatInput disabled` prop to include `keyGateBlocking`.

## TypeScript Typecheck

Command: `npm run check` (runs `tsc` from repo root)

**Baseline (before changes):**
```
client/src/lib/analyteParse.ts(11,18): error TS2307: Cannot find module 'papaparse' or its corresponding type declarations.
```
One pre-existing error — missing `@types/papaparse`, unrelated to this task.

**After changes:**
```
client/src/lib/analyteParse.ts(11,18): error TS2307: Cannot find module 'papaparse' or its corresponding type declarations.
```
Same single pre-existing error. Zero new errors introduced by this task.

## Self-Review

- Key is never written to `localStorage`, `sessionStorage`, cookies, or any URL parameter. Verified by inspection — only `useState`.
- `set_key` is sent in `onopen` before the client can call `sendMessage` (the browser cannot send user input before the socket fires `onopen`).
- The `needsKey` flag is set by `code === "NEEDS_KEY"`, exactly matching the ground-truth WS protocol in the brief (not a `needs_key` type field).
- `key_source` is handled in the `switch` — TypeScript's exhaustive narrowing covers the new union member.
- `ApiKeyGate` is suppressed behind `!hasAuthError` so a Clerk auth failure doesn't create conflicting UI states.
- The `keyGateBlocking` path blocks send but does NOT prevent the user from reading the existing chat — only the composer is disabled.

## Live Browser Verification (pending — manual step)

The following scenario should be verified in a running browser session:

```
1. Start backend with BYOK feature enabled and no server key configured for the test user.
2. Open the chat in the browser as a non-phenome (untrusted) user.
3. Type a message and press Send.
   Expected: server returns NEEDS_KEY error frame; chat shows an error bubble;
             amber ApiKeyGate form appears; composer textarea is disabled.
4. Paste a valid Anthropic API key into the gate form and click "Validate".
   Expected: form disappears; "clear key" affordance appears;
             composer is re-enabled; badge not yet visible.
5. Send a message.
   Expected: server sends key_source:"byok" before the response stream;
             badge reads "Using your key" in the mode toggle row;
             the response streams normally.
6. Click "Clear key".
   Expected: key is cleared; "clear key" affordance disappears; gate re-shows
             only on next NEEDS_KEY (badge clears on next turn).
7. Reload the page.
   Expected: key is gone (no persistence); no gate shown until another send attempt.
```
