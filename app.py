from fastapi import FastAPI
from pydantic import BaseModel
import gymnasium as gym

# Import your environment
from env.umbra_env import UmbraEnv 

app = FastAPI(title="UMBRA ShadowWorld API")

# Initialize the environment
umbra_env = UmbraEnv()

class ActionRequest(BaseModel):
    action: int

@app.get("/")
def read_root():
    return {"message": "UMBRA ShadowWorld Environment is running!"}

@app.post("/reset")
def reset_env():
    obs, info = umbra_env.reset()
    # Convert numpy arrays to lists for JSON serialization if needed
    if hasattr(obs, "tolist"):
        obs = obs.tolist()
    return {"observation": obs, "info": info}

@app.post("/step")
def step_env(req: ActionRequest):
    obs, reward, terminated, truncated, info = umbra_env.step(req.action)
    # Convert numpy arrays to lists for JSON serialization if needed
    if hasattr(obs, "tolist"):
        obs = obs.tolist()
    return {
        "observation": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }