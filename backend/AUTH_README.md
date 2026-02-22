# Authentication and Authorization

This document describes the authentication system added to the KRAKEN backend.

## Overview

The authentication system provides JWT-based authentication for WebSocket connections and API key validation for REST endpoints. It is designed with a feature flag (`AUTH_ENABLED`) for safe rollout and maintains backwards compatibility with existing conversations.

## Features

- **JWT-based WebSocket authentication** - Tokens passed via query parameter
- **API key validation for REST endpoints** - Bearer token authentication
- **User management** - User accounts linked to API keys
- **Feature flag** - Enable/disable authentication without code changes
- **Backwards compatibility** - Existing conversations work without user_id

## Architecture

### Database Schema

**kraken_users table:**
```sql
CREATE TABLE kraken_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hash
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    last_active TIMESTAMP DEFAULT NOW() NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL
);
```

**kraken_conversations table (updated):**
```sql
ALTER TABLE kraken_conversations
ADD COLUMN user_id UUID REFERENCES kraken_users(id);
```

The `user_id` column is nullable for backwards compatibility with existing conversations created before authentication was enabled.

### Authentication Flow

#### REST Endpoints

1. Client sends request with `Authorization: Bearer <api_key>` header
2. `validate_api_key()` dependency extracts and validates the API key
3. API key is hashed and checked against `kraken_users.api_key_hash`
4. User's `last_active` timestamp is updated
5. Request proceeds if valid, returns 401 if invalid

#### WebSocket Connections

1. Client connects to `/ws/chat?token=<jwt_token>`
2. `validate_ws_token()` decodes and validates the JWT
3. User existence and `is_active` status are verified
4. Connection is accepted if valid, closed with code 4001 if invalid
5. User's `last_active` timestamp is updated

### Security Considerations

**Known Limitations:**
- JWT tokens in query strings are visible in server logs
- This is documented as an acceptable trade-off for WebSocket authentication
- Production deployments should use HTTPS and secure log storage

**Best Practices:**
- API keys are hashed with SHA256 before storage
- JWT tokens expire after 7 days by default (configurable)
- Inactive users are rejected during authentication
- Database pool failures gracefully bypass authentication

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Authentication settings
AUTH_ENABLED=false                          # Feature flag (default: false)
JWT_SECRET_KEY=your-secret-key-here         # Change in production!
JWT_ALGORITHM=HS256                         # Default: HS256
JWT_EXPIRE_MINUTES=10080                    # 7 days (default: 10080)
API_KEYS=key1,key2,key3                     # Comma-separated list
```

### Frontend Configuration

Add to `client/.env`:

```bash
VITE_AUTH_TOKEN=<jwt_token>  # Optional: JWT token for authenticated connections
```

## Usage

### Creating Users

```python
from kestrel_backend.auth import create_user

# Create a new user with an API key
api_key = "my-secure-api-key"
user_id = await create_user(api_key)
```

### Generating JWT Tokens

```python
from kestrel_backend.auth import create_access_token
from datetime import timedelta

# Create a token for a user
token = create_access_token(
    data={"sub": user_id},
    expires_delta=timedelta(days=7)
)
```

### Protecting REST Endpoints

```python
from fastapi import Depends
from kestrel_backend.auth import validate_api_key

@app.get("/api/protected")
async def protected_endpoint(auth: dict = Depends(validate_api_key)):
    user_id = auth["user_id"]
    # ... endpoint logic
```

### Client-Side Authentication

The frontend automatically appends the token to WebSocket URLs if `VITE_AUTH_TOKEN` is set:

```typescript
// useWebSocket.ts automatically handles this
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN || "";
const wsUrlWithAuth = AUTH_TOKEN ? `${WS_URL}?token=${AUTH_TOKEN}` : WS_URL;
```

On authentication failure (close code 4001), the connection status changes to `auth_failed` and displays an error message.

## Testing

### Running Tests

```bash
cd backend
uv run pytest tests/test_auth.py -v
```

### Test Coverage

- API key hashing and verification
- JWT token creation and validation
- REST endpoint authentication (with mocked database)
- WebSocket token validation (with mocked database)
- User creation and duplicate detection
- Inactive user rejection
- Expired token handling
- Missing token handling

## Deployment

### Safe Rollout Strategy

1. **Deploy with `AUTH_ENABLED=false`**
   ```bash
   AUTH_ENABLED=false  # Start with auth disabled
   ```

2. **Run migrations**
   ```bash
   alembic upgrade head  # Creates kraken_users table
   ```

3. **Create test users**
   ```python
   # Create users for testing
   await create_user("test-api-key-1")
   await create_user("test-api-key-2")
   ```

4. **Test with auth enabled**
   ```bash
   AUTH_ENABLED=true
   API_KEYS=test-api-key-1,test-api-key-2
   ```

5. **Verify WebSocket connections work** with valid tokens

6. **Enable in production**
   ```bash
   AUTH_ENABLED=true
   JWT_SECRET_KEY=<strong-random-secret>
   API_KEYS=<production-keys>
   ```

### Production Checklist

- [ ] Generate strong JWT secret key (32+ characters)
- [ ] Create production API keys (use cryptographically secure random strings)
- [ ] Store API keys securely (password manager, secrets manager)
- [ ] Enable HTTPS for production deployment
- [ ] Configure secure log storage (to protect token visibility)
- [ ] Set up user creation process/workflow
- [ ] Document token distribution process for users
- [ ] Test auth failure scenarios
- [ ] Monitor `last_active` timestamps for user activity

## API Reference

### `auth.py` Module

#### `hash_api_key(api_key: str) -> str`
Hash an API key using SHA256.

#### `verify_api_key(api_key: str, hashed_key: str) -> bool`
Verify an API key against its hash.

#### `create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str`
Create a JWT access token.

#### `validate_api_key(credentials: HTTPAuthorizationCredentials) -> dict`
FastAPI dependency for validating API keys on REST endpoints.

Returns: `{"authenticated": bool, "user_id": str | None}`

Raises: `HTTPException(401)` if authentication fails.

#### `validate_ws_token(token: Optional[str]) -> Optional[dict]`
Validate a JWT token for WebSocket connections.

Returns: `{"user_id": str}` if valid, `None` if auth disabled.

Raises: `ValueError` if token is invalid or expired.

#### `create_user(api_key: str) -> str`
Create a new user with an API key.

Returns: User ID (UUID as string)

Raises: `ValueError` if creation fails or key exists.

## Migration Details

**Migration 002_add_users_table.py:**
- Creates `kraken_users` table
- Adds `user_id` column to `kraken_conversations` (nullable)
- Creates necessary indexes and foreign keys
- Uses `IF NOT EXISTS` for idempotency

## Troubleshooting

### WebSocket connection closes immediately

Check for close code 4001 - this indicates authentication failure. Verify:
- Token is valid and not expired
- User exists and is active
- `AUTH_ENABLED=true` in environment

### REST endpoint returns 401

Verify:
- API key is included in `Authorization: Bearer <key>` header
- API key exists in database with matching hash
- User is active (`is_active=true`)

### Database pool not initialized

If you see "Database pool not initialized" warnings:
- Check `DATABASE_URL` environment variable
- Verify database is running and accessible
- Authentication will be bypassed if pool is unavailable

### Migrations fail

If migrations fail on startup:
- Check that alembic is installed: `uv sync`
- Verify `DATABASE_URL` is correct
- Ensure database has proper permissions
- Migrations are non-fatal - app will start anyway

## Future Enhancements

Potential improvements for future iterations:

1. **Role-based access control (RBAC)** - Add user roles and permissions
2. **Refresh tokens** - Implement token refresh mechanism
3. **Token revocation** - Add blacklist for revoked tokens
4. **Rate limiting per user** - Track and limit requests by user_id
5. **OAuth2 integration** - Support third-party authentication
6. **API key rotation** - Allow users to regenerate keys
7. **Multi-factor authentication** - Add 2FA support
8. **Audit logging** - Track all authentication events
