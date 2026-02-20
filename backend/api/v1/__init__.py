from fastapi import APIRouter
from . import auth, images, styles, filters, health

router = APIRouter(prefix="/v1")

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(images.router, prefix="/images", tags=["images"])
router.include_router(styles.router, prefix="/styles", tags=["styles"])
router.include_router(filters.router, prefix="/filters", tags=["filters"])
router.include_router(health.router, tags=["health"])

__all__ = ["router"]
