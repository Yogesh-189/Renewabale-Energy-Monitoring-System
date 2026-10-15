"""
Security - Stage 8
-------------------
Protects the API with a login. Uses only the Python standard library, so there
is nothing extra to install.

HOW IT WORKS
  1. The dashboard asks GET /auth/status whether a login is required.
  2. The user signs in: POST /auth/login with username + password.
  3. The server answers with a signed token (a JWT, HS256) that expires after a
     few hours. The browser sends it as  Authorization: Bearer <token>  on every
     later request. Anything without a valid token gets 401 Unauthorized.
  4. Repeated wrong passwords are throttled (HTTP 429), which blocks guessing.
  5. Every response carries browser-safety headers; over HTTPS it also tells
     browsers to never fall back to plain HTTP (HSTS).

TURNING IT ON (environment variables on the server)
  AUTH_PASSWORD     required to enable the login - choose a long one
  AUTH_USERNAME     optional, default 'admin'
  AUTH_SECRET       secret used to sign tokens; set a long random value
                    (python -c "import secrets; print(secrets.token_hex(32))")
  AUTH_TOKEN_HOURS  optional, how long a login lasts, default 8

If AUTH_PASSWORD is not set the API stays open, which is convenient on your own
laptop. On a public server ALWAYS set it.

HONEST LIMITS: one shared account (no user database or roles); tokens cannot be
revoked before they expire, so 'sign out' only forgets the token in the browser;
the password lives in an environment variable. All reasonable for this project,
and each is a clear 'future work' item.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Optional, Tuple

# Reachable without a token. Everything else is protected automatically,
# including any endpoint added in the future.
PUBLIC_PATHS = {"/", "/health", "/auth/login", "/auth/status"}


# ---------------------------------------------------------------- tokens (JWT HS256)
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_token(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), f"{h}.{p}".encode("ascii"), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


def verify_token(token: str, secret: str, now: Optional[float] = None) -> Optional[dict]:
    """Returns the token's payload if it is genuine and unexpired, else None."""
    try:
        h, p, s = token.split(".")
        header = json.loads(_b64url_decode(h))
        if header.get("alg") != "HS256":           # refuse 'none' and any other algorithm
            return None
        expected = hmac.new(secret.encode("utf-8"), f"{h}.{p}".encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url(expected), s):
            return None
        payload = json.loads(_b64url_decode(p))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)) or (time.time() if now is None else now) >= exp:
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------- configuration
class AuthConfig:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.username = env.get("AUTH_USERNAME", "admin")
        self.password = env.get("AUTH_PASSWORD", "")
        self.enabled = bool(self.password)
        secret = env.get("AUTH_SECRET", "")
        self.secret_is_random = not secret
        self.secret = secret or secrets.token_hex(32)
        try:
            self.token_hours = float(env.get("AUTH_TOKEN_HOURS", "8"))
        except ValueError:
            self.token_hours = 8.0

    def describe(self):
        """Lines to print at start-up so you can see the security state."""
        if not self.enabled:
            return ["Login DISABLED - the API is open. Set AUTH_PASSWORD to enable it."]
        lines = [f"Login ENABLED for user '{self.username}' (tokens last {self.token_hours:g} h)."]
        if len(self.password) < 10:
            lines.append("WARNING: AUTH_PASSWORD is short. Use at least 10 characters.")
        if self.secret_is_random:
            lines.append("AUTH_SECRET not set: using a random one, so everyone must sign in again "
                         "after each server restart. Set AUTH_SECRET for stable logins.")
        return lines

    def check_credentials(self, username: str, password: str) -> bool:
        # Both comparisons always run, in constant time, so timing reveals nothing.
        user_ok = hmac.compare_digest(str(username).encode("utf-8"), self.username.encode("utf-8"))
        pass_ok = hmac.compare_digest(str(password).encode("utf-8"), self.password.encode("utf-8"))
        return user_ok and pass_ok

    def issue_token(self, username: str) -> Tuple[str, int]:
        now = int(time.time())
        exp = now + int(self.token_hours * 3600)
        return create_token({"sub": username, "iat": now, "exp": exp}, self.secret), exp

    def verify(self, token: str) -> Optional[dict]:
        return verify_token(token, self.secret)


def authorize(config: AuthConfig, path: str, bearer_token: Optional[str]) -> Tuple[bool, str]:
    """Decides whether a request may proceed. Returns (allowed, reason)."""
    if not config.enabled or path in PUBLIC_PATHS:
        return True, "public"
    if not bearer_token:
        return False, "Login required"
    if config.verify(bearer_token) is None:
        return False, "Session expired or invalid - please sign in again"
    return True, "ok"


# ---------------------------------------------------------------- brute-force throttle
class LoginThrottle:
    """Blocks password guessing: at most `per_client` failures per client and
    `overall` failures from everyone within `window_seconds`. The overall cap
    still holds if an attacker fakes their address."""

    def __init__(self, per_client: int = 5, overall: int = 30, window_seconds: int = 300):
        self.per_client, self.overall, self.window = per_client, overall, window_seconds
        self._by_client = {}
        self._all = []
        self._lock = threading.Lock()

    def _prune(self, now):
        cutoff = now - self.window
        self._all = [t for t in self._all if t > cutoff]
        for key in list(self._by_client):
            self._by_client[key] = [t for t in self._by_client[key] if t > cutoff]
            if not self._by_client[key]:
                del self._by_client[key]

    def retry_after(self, client: str) -> int:
        """Seconds the client must wait; 0 means the attempt is allowed."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            waits = []
            mine = self._by_client.get(client, [])
            if len(mine) >= self.per_client:
                waits.append(self.window - (now - mine[0]))
            if len(self._all) >= self.overall:
                waits.append(self.window - (now - self._all[0]))
            return int(max(waits)) + 1 if waits else 0

    def record_failure(self, client: str):
        with self._lock:
            now = time.monotonic()
            self._by_client.setdefault(client, []).append(now)
            self._all.append(now)

    def reset(self, client: str):
        with self._lock:
            self._by_client.pop(client, None)


def client_ip(forwarded_for: Optional[str], direct: Optional[str]) -> str:
    """Behind a proxy (Render) the real client is the LAST address the proxy
    appended to X-Forwarded-For; earlier entries can be forged by the sender."""
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return direct or "unknown"


def attempt_login(config: AuthConfig, throttle: LoginThrottle, username: str,
                  password: str, client: str) -> Tuple[int, dict]:
    """Returns (http_status, body). Keeps all the login rules in one testable place."""
    if not config.enabled:
        return 200, {"auth_required": False, "access_token": None,
                     "detail": "Login is not enabled on this server."}
    wait = throttle.retry_after(client)
    if wait:
        return 429, {"detail": f"Too many failed attempts. Try again in {wait} seconds.",
                     "retry_after": wait}
    if not config.check_credentials(username, password):
        throttle.record_failure(client)
        return 401, {"detail": "Wrong username or password."}
    throttle.reset(client)
    token, exp = config.issue_token(username)
    return 200, {"access_token": token, "token_type": "bearer", "expires_at": exp,
                 "username": username}


# ---------------------------------------------------------------- response headers
def security_headers(scheme: str, path: str) -> dict:
    headers = {
        "X-Content-Type-Options": "nosniff",      # browser must not guess file types
        "X-Frame-Options": "DENY",                # page cannot be embedded in another site
        "Referrer-Policy": "no-referrer",
    }
    if scheme == "https":
        headers["Strict-Transport-Security"] = "max-age=31536000"   # always use HTTPS next time
    if not path.startswith("/dashboard"):
        headers["Cache-Control"] = "no-store"     # API answers are never cached
    return headers
