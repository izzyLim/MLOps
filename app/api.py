from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from .schemas import Wine, Rating, feature_names

from .monitoring import instrumentator


ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
model = load(ROOT_DIR / "artifacts/model.joblib")

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


@app.get("/")
def root():
    return "Wine Quality Ratings"

@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine):
    sample_dict = sample.dict()
    features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    response.headers["X-model-score"] = str(prediction)
    return Rating(quality=prediction)


from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

@app.post("/sa_predict")
def sa_predict(response: Response):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # 모델과 tokenizer 객체를 사용하여 pipeline 초기화
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 예측
    result = classifier("I love using transformers!")
    response.headers["X-model-accuracy"] = "0.98"
    print(result)
    return {"result": result}

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}