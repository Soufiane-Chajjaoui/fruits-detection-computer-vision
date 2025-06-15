from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from controllers.object_detection_controller import router as detection_router
import os

# Créer le dossier static s'il n'existe pas
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="Object Detection API",
    description="API pour la détection d'objets utilisant YOLOv8",
    version="1.0.0",
    contact={
        "name": "Chajjaoui Soufiane",
        "email": "schajjaoui@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Monter le dossier static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inclure les routers des controllers
app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/realtime", response_class=HTMLResponse)
async def realtime_detection_page():
    """
    Page de démonstration pour la détection d'objets en temps réel.
    """
    with open("static/realtime-detection.html", "r") as f:
        html_content = f.read()
    return html_content
