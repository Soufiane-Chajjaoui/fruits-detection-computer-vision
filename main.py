from fastapi import FastAPI
from controllers.object_detection_controller import router as detection_router

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

# Inclure les routers des controllers
app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
