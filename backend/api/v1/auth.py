from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt

from backend.core.database import get_db
from backend.core.config import settings
from backend.models import User, RefreshToken
from backend.schemas import (
    RegisterRequest,
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {"sub": user_id, "exp": expire}
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": user_id, "exp": expire, "type": "refresh"}
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    # Check if username exists
    result = await db.execute(select(User).where(User.username == request.username))
    if result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Find user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalars().first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Create tokens
    access_token = create_access_token(str(user.id))
    refresh_token_str = create_refresh_token(str(user.id))

    # Store refresh token
    refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=refresh_token_str,
        expires_at=datetime.utcnow()
        + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_token)
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(
            request.refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    # Verify token in database
    result = await db.execute(
        select(RefreshToken).where(
            (RefreshToken.token_hash == request.refresh_token)
            & (RefreshToken.revoked_at.is_(None))
            & (RefreshToken.expires_at > datetime.utcnow())
        )
    )
    stored_token = result.scalars().first()

    if not stored_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    # Create new access token
    access_token = create_access_token(user_id)

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
