import os
import warnings
from datetime import datetime, timedelta

import bcrypt
import jwt                          # PyJWT
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# ── Settings ─────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-insecure-secret-change-me")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

if SECRET_KEY == "dev-insecure-secret-change-me":
    warnings.warn(
        "JWT_SECRET_KEY is not set — using an insecure dev default. "
        "Set the JWT_SECRET_KEY environment variable before deploying.",
        stacklevel=1,
    )

# ── Password helpers (bcrypt directly — passlib is unmaintained & incompatible
#    with bcrypt >= 4.0, so we call bcrypt directly instead) ──────────────────

def get_password_hash(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ── In-memory user store ──────────────────────────────────────────────────────
# TODO: replace with DB-backed user table (models.py → User).
# Hash computed once at startup — no module-level side effects.
USERS_DB: dict[str, dict] = {
    "admin1": {
        "username":      "admin1",
        "password_hash": get_password_hash("password"),
        "role":          "admin",
    },
}

# ── FastAPI helpers ───────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username:     str
    role:         str
    display_name: str = ""   # shown in UI; falls back to username if blank

def authenticate_user(username: str, password: str) -> dict | None:
    """
    Check credentials. SQL users table is the source of truth; USERS_DB is a
    fallback for the bootstrap admin account before the DB is seeded.
    """
    # 1) Try SQL users table first (covers all created-via-UI users)
    try:
        from database.session import SessionLocal
        from database.models import User as UserModel
        db = SessionLocal()
        try:
            row = db.query(UserModel).filter(UserModel.username == username).first()
            if row and verify_password(password, row.password_hash):
                return {"username": row.username, "role": row.role,
                        "display_name": row.display_name or row.username,
                        "password_hash": row.password_hash}
        finally:
            db.close()
    except Exception:
        pass  # DB not ready yet (e.g. first startup before init_db)

    # 2) Fall back to in-memory bootstrap dict
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user

# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload      = decode_token(token)
    username     = payload.get("sub")
    role         = payload.get("role")
    display_name = payload.get("display_name", "")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return User(username=username, role=role, display_name=display_name or username)

def require_role(allowed_roles: list[str]):
    def checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user
    return checker