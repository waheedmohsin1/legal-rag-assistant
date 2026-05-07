from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag_pipeline import ask_question

import uvicorn


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):
    question: str


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    # return templates.TemplateResponse(
    #     "index.html",
    #     {"request": request}
    # )
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =========================================================
# ASK ENDPOINT
# =========================================================

@app.post("/ask")
async def ask(data: QuestionRequest):

    result = ask_question(data.question)

    return result


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=4587,
        reload=True
    )