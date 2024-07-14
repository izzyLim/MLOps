import datetime
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
import random
from datetime import datetime, timedelta

with DAG(
    "test_dag",
    schedule=timedelta(days=30),
    start_date=pendulum.datetime(2023, 8, 1, tz="UTC"),
    catchup=False,
) as dag:
    
    def select_dog():
        dogs = ['Bulldog','Pomeranian','Poodle','Chihuahua','Dashhund']
        rand_int = random.randint(0,4)
        print(dogs[rand_int])

    py_t1 = PythonOperator(
        task_id = 'py_t1',
        python_callable=select_dog
    )
    py_t1
