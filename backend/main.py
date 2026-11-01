from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.intake import router as intake_router

app = FastAPI(
    title="Nirvor - A Neuro-Symbolic Engine",
    version="2.0.0",
    description="Meta Global AI Developer Hackathon Submission - Team Anindo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(intake_router)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Neuro-Symbolic Bayesian OCR & Regional Voice Fusion",
        "team": "Team Anindo",
    }
