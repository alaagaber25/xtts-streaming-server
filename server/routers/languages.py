from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/languages")
def get_languages(request: Request):
    return request.app.state.xtts_config.languages

