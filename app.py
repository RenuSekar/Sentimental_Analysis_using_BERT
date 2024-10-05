from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sentimental_model

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("./static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/analyze-sentiment")
async def analyze_sentiment(request: TextRequest):
    score = sentimental_model.sentiment_score(request.text)  
    return {"sentiment_score": score}  