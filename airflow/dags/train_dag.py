from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

from tasks.train import train
import pandas as pd
import plotly
#import tensorflow
#import numpy as np
#import sklearn
#import kaleido



with DAG(
    "train",
    default_args={
        "depends_on_past": False,
        #"retries": 1,
        #"retry_delay": timedelta(minutes=5),
    },
    description="train",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False,
) as dag:
    train_task = PythonOperator(
        task_id="train_task",
        #requirements=["plotly", "scikit-learn", "tensorflow", "numpy", "pandas", "kaleido"],
        python_callable=train
    )
    train_task