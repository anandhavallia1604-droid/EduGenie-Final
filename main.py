from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from explanation_module import explain_topic
from learning_path import get_learning_recommendations
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


app = FastAPI(
    title="EduGenie",
    description="Google Gemini powered learning assistant",
    version="1.0.0",
)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=5000)


class QuizRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "EduGenie"}


@app.post("/qa")
async def qa(request: TextRequest):
    try:
        return {"answer": answer_question(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain")
async def explain(request: TopicRequest):
    try:
        return {
            "topic": request.topic,
            "explanation": explain_topic(request.topic),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/summarize")
async def summarize(request: TextRequest):
    try:
        return {"summary": summarize_text(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/quiz")
async def quiz(request: QuizRequest):
    try:
        return {"quiz": generate_quiz(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/learn/recommendations")
async def recommendations(request: TopicRequest):
    try:
        return {
            "topic": request.topic,
            "recommendation": get_learning_recommendations(request.topic),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return await _json_error(400, str(exc))


async def _json_error(status_code: int, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )