from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api import router

app = FastAPI(title="Fitness AI Agent")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Mount API under /api
app.include_router(router, prefix="/api")


@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/app")
def app_main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})
