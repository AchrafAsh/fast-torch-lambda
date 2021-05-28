from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def testing_child_resoure():
    return {"message": "This is a test endpoint!"}