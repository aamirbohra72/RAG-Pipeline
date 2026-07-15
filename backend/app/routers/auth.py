from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import AuthResponse, LoginRequest, RegisterRequest, UserOut
from app.services import auth_service
from app.services.auth_service import User, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    try:
        user = auth_service.create_user(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = auth_service.create_access_token(user)
    return AuthResponse(
        access_token=token,
        user=UserOut(id=user.id, email=user.email),
    )


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    user = auth_service.authenticate(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = auth_service.create_access_token(user)
    return AuthResponse(
        access_token=token,
        user=UserOut(id=user.id, email=user.email),
    )


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email)
