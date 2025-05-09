# backend/main.py
from fastapi import FastAPI
from backend import db                       # ← تأكد من المسار
from sqlmodel import Session

app = FastAPI()                              # يجب أن يكون بهذا الاسم بالضبط

db.init_db()

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/cv/{cv_id}")
def get_cv(cv_id: int):
    with Session(db.engine) as session:
        cv = db.get_cv(session, cv_id)
        if not cv:
            return {"error": "CV not found"}
        return cv
    # return {"cv": cv}
    