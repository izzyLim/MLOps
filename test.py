import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os


EXPERIMENT_NAME = "test_240629_02"

#ARTIFACT_ROOT = os.getenv('ARTIFACT_ROOT')
#MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'


if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():        
        mlflow.sklearn.autolog()
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        print(f"Score: {score}")
        mlflow.log_metric("score", score)
        predictions = lr.predict(X)
        signature = infer_signature(X, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)
        print(f"Model saved in run {mlflow.active_run().info.run_uuid}")