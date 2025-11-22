from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from analytics.events import log_event
from db.mongo import get_database
from .hash import hash_password, verify_password
from .jwt_handler import create_access_token, get_current_user


router = APIRouter(prefix="/auth", tags=["auth"])


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: EmailStr
    createdAt: datetime


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_in: UserCreate) -> TokenResponse:
    db = get_database()
    users = db["users"]

    existing = users.find_one({"email": user_in.email})
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    password_hash = hash_password(user_in.password)
    now = datetime.utcnow()

    user_doc: Dict[str, Any] = {
        "email": user_in.email,
        "password_hash": password_hash,
        "createdAt": now,
    }

    result = users.insert_one(user_doc)
    user_id = str(result.inserted_id)

    access_token = create_access_token(user_id)
    return TokenResponse(access_token=access_token)


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin) -> TokenResponse:
    db = get_database()
    users = db["users"]

    user = users.find_one({"email": credentials.email})
    if user is None or not verify_password(credentials.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user_id = str(user["_id"])
    access_token = create_access_token(user_id)

    # Analytics: track successful logins
    log_event("user_login", user_id=user_id)

    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> UserOut:
    return UserOut(
        id=current_user.get("id", str(current_user.get("_id"))),
        email=current_user["email"],
        createdAt=current_user["createdAt"],
    )
