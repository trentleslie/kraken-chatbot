# Task: Set Up Dev Branch Deployment Workflow

## Goal
Create a dev branch deployment workflow for kraken-chatbot that mirrors the biomapper-ui pattern. When code is pushed to a `dev` branch, it should auto-deploy to `dev-kraken.expertintheloop.io`.

## Context
- **Production** deploys from `main` → `kraken.expertintheloop.io` (port 8000)
- **Dev** should deploy from `dev` → `dev-kraken.expertintheloop.io` (port **8006**)
- **Server:** AWS Lightsail `expert-in-the-loop-upgraded` (IP may change on instance cycle — use DNS, don't hardcode)
- **SSH:** `ssh lightsail` (alias in `~/.ssh/config`, uses `~/.ssh/lightsail-expert.pem`)
- **OS:** Ubuntu 22.04.5, Node v20.20.0, Python 3.10.12 (system)
- **Service:** `kraken-backend` (systemd) for prod, `kraken-backend-dev` for dev
- **All services run as user `ubuntu`** — shared user across all services
- **Database:** PostgreSQL 16 on 127.0.0.1:5432 (shared, used by kraken)
- **Reverse proxy:** nginx (80/443) with Let's Encrypt via certbot

### PORT MAP — DO NOT CONFLICT
These ports are already in use on the server:
| Port | Service |
|------|---------|
| 5000 | expert-in-the-loop |
| 5432 | PostgreSQL 16 |
| 8000 | **kraken-backend (PROD)** |
| 8001 | biomapper2-api |
| 8002 | biomapper-ui-python |
| 8003 | dev-biomapper |
| 8080 | biomapper-ui-express |
| 8501 | pgs-catalog-explorer |
| 8502 | kraken-analytics |

**Available ports:** 8004-8007, 8100+, 9000+
**Chosen dev port: 8006** (8004/8005 reserved for biomapper-ui dev)

### Reference pattern
biomapper-ui has the proven dev branch workflow. The repo may be at:
- `~/trentleslie@gmail.com/Google Drive/projects/biomapper-ui/`
- Or search for it: `find ~ -maxdepth 4 -type d -name "biomapper-ui" 2>/dev/null`

Study its `deploy/dev/`, `.github/workflows/deploy-dev.yml`, and nginx configs.

## Architecture (single-stack, simpler than biomapper-ui)
- **Frontend:** React 18 + Vite (built to `dist/`) — `VITE_WS_URL` baked at build time
- **Backend:** FastAPI + Uvicorn on a single port
- **WebSocket:** `/ws/chat` endpoint needs proxy with upgrade headers
- **Database:** PostgreSQL 16 with Alembic migrations
- **Dependencies:** `uv` for Python, `npm` for frontend
- **No Docker, no PM2** — everything runs via systemd

## Steps

### 1. Study the reference pattern
Read these files from biomapper-ui to understand the proven pattern:
- `.github/workflows/deploy-dev.yml`
- `deploy/dev/` (all files)
- `.github/workflows/deploy.yml` (prod, for comparison)

Also read this repo's existing `.github/workflows/deploy.yml` to understand the current prod deployment.

### 2. Handle existing branches
- There's an existing `origin/dev_main` branch — check if it has meaningful divergence from `main`. If it's stale/abandoned, note it but don't delete it (leave that for Trent).
- Create the `dev` branch from current `main`: `git checkout -b dev`

### 3. Create deployment infrastructure

**Create `deploy/` directory** with these files:

#### `deploy/kraken-backend.service`
Production systemd service (PORT=8000). Version-control what's already running on the server. Model after the existing deployment in `.github/workflows/deploy.yml`.

**CRITICAL — ProtectHome hazard:** If you use `ProtectHome=read-only`, you MUST add cache paths to `ReadWritePaths`. All services share the `ubuntu` user, and making `~/.cache/uv/` read-only will crash OTHER services (this has happened before). Include at minimum:
```ini
ReadWritePaths=/home/ubuntu/kraken-chatbot /home/ubuntu/.cache/uv /tmp
```
Or better yet, skip `ProtectHome` entirely to avoid cross-service breakage.

#### `deploy/nginx-kraken.conf`
Production nginx config for `kraken.expertintheloop.io`. Include:
- SPA fallback (`try_files $uri $uri/ /index.html`)
- WebSocket proxy at `/ws/chat` with upgrade headers and 300s timeout
- API proxy at `/api/`
- Health endpoint at `/health`
- Static files from `/home/ubuntu/kraken-chatbot/dist/public`

#### `deploy/dev/kraken-backend-dev.service`
Dev systemd service — identical to prod but:
- **`PORT=8006`** (not 8000, and NOT 8001 which is biomapper2-api)
- `WorkingDirectory` points to `/home/ubuntu/kraken-chatbot-dev`
- Service name: `kraken-backend-dev`
- Use `$DEPLOY_DIR` as a placeholder that the deploy script replaces with `sed`
- Same ProtectHome caution as above

#### `deploy/dev/nginx-kraken-dev.conf`
Dev nginx config — identical to prod but:
- `server_name dev-kraken.expertintheloop.io`
- Proxy to port **8006**
- Root at `/home/ubuntu/kraken-chatbot-dev/dist/public`
- WebSocket proxy to port **8006**

#### `deploy/dev/.env.example`
```
HOST=127.0.0.1
PORT=8006
ALLOWED_ORIGINS=https://dev-kraken.expertintheloop.io
RATE_LIMIT_PER_MINUTE=20
KESTREL_API_KEY=
DATABASE_URL=
AUTH_ENABLED=false
```

### 4. Create GitHub Actions dev workflow

**Create `.github/workflows/deploy-dev.yml`**

Model closely after biomapper-ui's deploy-dev.yml but adapted for kraken's simpler architecture:
- **Trigger:** push to `dev` branch + `workflow_dispatch` with optional branch input
- **Concurrency:** `lightsail-deploy-kraken-dev`, cancel-in-progress: false
- **Environment:** `development`
- **Deploy dir:** `/home/ubuntu/kraken-chatbot-dev`
- **Build:** `VITE_WS_URL=wss://dev-kraken.expertintheloop.io/ws/chat npm run build`
- **Backend:** `cd backend && uv sync`, then alembic migrations
- **Service restart:** `sudo systemctl restart kraken-backend-dev`
- **Health check:** retry loop on `http://localhost:8006/ready`

**Safety:** Always run `sudo nginx -t` before `sudo systemctl reload nginx` in any deploy script.

Use the same SSH action and secrets pattern as the existing `deploy.yml`.

### 5. Update existing deploy.yml
Add path-based triggers to the existing production workflow if not already present:
```yaml
paths:
  - 'client/**'
  - 'backend/**'
  - 'package.json'
  - '.github/workflows/deploy.yml'
```

### 6. Update CLAUDE.md
Add a "Dev Branch Workflow" section documenting:
- Dev URL: `dev-kraken.expertintheloop.io`
- Dev port: 8006
- How to create feature branches from dev
- Manual deployment steps
- The relationship between dev and main branches

### 7. Update backend ALLOWED_ORIGINS
In `backend/src/kestrel_backend/main.py`, check how CORS origins are configured. Make sure it reads from an environment variable so dev and prod can have different origins without code changes.

### 8. Commit and push
- Commit all changes to the `dev` branch with a clear message
- Push with `git push -u origin dev`
- Do NOT merge to main — this stays on dev

## Important Notes
- Do NOT SSH into the server or make any server-side changes — only create the files and configs that will be deployed via GitHub Actions
- The server-side setup (cloning the dev repo, enabling nginx config, creating .env, certbot for SSL) will be done manually by Trent
- Keep the pattern as close to biomapper-ui as possible for consistency
- The `$DEPLOY_DIR` placeholder in service files should match what the deploy script uses via `sed`
- Check if GitHub secrets for `LIGHTSAIL_HOST`, `LIGHTSAIL_USER`, `LIGHTSAIL_SSH_KEY` already exist (they should, since prod deploys work)
- **Never hardcode the Lightsail IP** — it can change on instance cycle
- **All services share user `ubuntu`** — systemd security directives on one service can break others

## Server-Side Setup (for Trent, after agent completes)
These steps happen manually on the server after the agent creates all the files:
1. SSH in: `ssh lightsail`
2. Clone dev copy: `git clone <repo> /home/ubuntu/kraken-chatbot-dev && cd /home/ubuntu/kraken-chatbot-dev && git checkout dev`
3. Create `.env` from the example
4. Install the systemd service: `sudo cp deploy/dev/kraken-backend-dev.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable kraken-backend-dev`
5. Install nginx config: `sudo cp deploy/dev/nginx-kraken-dev.conf /etc/nginx/sites-available/ && sudo ln -s /etc/nginx/sites-available/nginx-kraken-dev.conf /etc/nginx/sites-enabled/`
6. Get SSL cert: `sudo certbot --nginx -d dev-kraken.expertintheloop.io`
7. Test and reload: `sudo nginx -t && sudo systemctl reload nginx`
8. Start the service: `sudo systemctl start kraken-backend-dev`

## Verification
Before committing, verify:
- [ ] `deploy/` directory has both prod and dev configs
- [ ] `.github/workflows/deploy-dev.yml` exists and references port **8006** everywhere
- [ ] No references to port 8001 anywhere (that's biomapper2-api)
- [ ] CORS configuration supports environment-based origins
- [ ] CLAUDE.md documents the dev workflow
- [ ] All file paths in deploy scripts match the actual repo structure
- [ ] WebSocket proxy config includes upgrade headers
- [ ] nginx configs use `sudo nginx -t` before reload
- [ ] systemd services don't use `ProtectHome` without proper `ReadWritePaths`
- [ ] Deploy scripts don't hardcode the Lightsail IP

## Clean up
Delete this file (AGENT-TASK-dev-branch.md) after the task is complete — it shouldn't be committed to the repo.
