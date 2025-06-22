from fastapi import FastAPI
from routes import user
app = FastAPI()
app.include_router(user.router)


@app.get("/")
async def homepage():
    return {"data": "welcome to the application"}

