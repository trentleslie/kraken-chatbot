# UI Overhaul: Phenome Health Design System

## Objective

Overhaul the KRAKEN Chatbot frontend to conform to the Phenome Health design system. Every component, layout, and visual decision should match the design system spec — the goal is that this app looks like it came from the same organization as every other Phenome Health tool.

## Setup

1. **Create a feature branch off `dev`:**
   ```
   git checkout dev && git pull origin dev
   git checkout -b feat/phenome-ui-overhaul
   ```

2. **Load the design system:**
   Invoke `/phenome-ui` to load the full Phenome Health design system into context. This is mandatory before making any UI changes.

3. **Locate the frontend:**
   The frontend source is at `client/src/`. Key directories:
   - `components/` — reusable UI components
   - `pages/` — route-level page components
   - `hooks/` — custom React hooks
   - `lib/` — utility functions
   - `types/` — TypeScript type definitions
   - `utils/` — utility functions
   - `index.css` — global styles
   - `App.tsx` — root component and routing

## Scope

### Must Do
- Apply the Phenome color tokens (`ph-navy`, `ph-crimson`, `ph-teal`, `ph-green`, neutrals) — replace any ad-hoc colors
- Switch typography to Inter (body) / JetBrains Mono (IDs, code) with the design system's type scale (14px body default)
- Apply the layout shell: TopBar (h-14) + SideNav (w-60) + content area (max-w-screen-2xl) — adapt for the chat interface where the main content area may need a different layout to accommodate the conversation panel
- Restyle all buttons to the 5-variant system (primary, secondary, ghost, destructive, link)
- Restyle cards to white bg + 1px neutral-200 border, no shadow (shadow-xs on hover only)
- Restyle tables to match §6.3 (uppercase column headers, tabular-nums, hover rows)
- Restyle form inputs to match §5.2 (neutral-300 border, navy focus ring)
- Restyle the chat interface: message bubbles, input area, and any conversation UI should use the design system tokens while remaining functional as a chat experience
- Add/fix loading (skeleton), empty, and error states for all async surfaces
- Set border-radius to `rounded` (4px) default — remove any `rounded-xl`, `rounded-2xl`, `rounded-3xl`
- Remove any gradients, glassmorphism, or decorative animations
- Ensure accessibility: focus rings, ARIA labels, color-not-only signifiers

### Must NOT Do
- Do not change routing, data fetching logic, WebSocket connections, or API integration
- Do not add new dependencies beyond what the design system requires (Inter font, JetBrains Mono font)
- Do not restructure the component file organization
- Do not modify the FastAPI backend (`backend/`) or the LangGraph pipeline

### Apply the Tailwind Config
Update the Tailwind config with the token extensions from §3.6 of the design system. Extend, don't override, any existing config.

### Apply the Global CSS
Update `client/src/index.css` with the base layer styles from §3.7 (Inter font import, tabular-nums on tables, focus-visible ring).

## Workflow

1. Start with the Tailwind config and global CSS — these affect everything
2. Build the app shell (TopBar, SideNav) if not already present, or restyle the existing one
3. Restyle the chat interface — this is the core experience; apply tokens but preserve usability
4. Work through remaining components: buttons → inputs → cards → badges → alerts → modals → tables
5. Update each page to use the restyled components
6. Run the component checklist (design system §11) on each component before moving on
7. Test at `lg:` breakpoint (1024px) — this is the primary target. Graceful degradation for smaller screens.

## Commit Strategy

Commit after each logical chunk (e.g., "Apply Phenome tokens and global CSS", "Restyle app shell", "Restyle chat interface"). Keep commits on the `feat/phenome-ui-overhaul` branch. Do not merge to `dev` — open a PR when complete.
