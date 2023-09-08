import os

import pandas as pd
from sqlalchemy import create_engine

import mlflow

def monitor_with_io(predictions_folder: str, db_con_str: str, monitoring_table_name: str,
                    model_name:str) -> None:
    latest_predictions_path = os.path.join(predictions_folder, 'latest.csv')
    latest_predictions = pd.read_csv(latest_predictions_path,
                                     usecols=['predictions_time', 'predictions'],
                                     parse_dates=['predictions_time'],
                                     date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d-%H%M%S'))

    monitoring_df = monitor(latest_predictions, model_name)

    engine = create_engine(db_con_str)
    db_conn = engine.connect()
    monitoring_df.to_sql(monitoring_table_name, con=db_conn, if_exists='append', index=False)
    db_conn.close()


def monitor(latest_predictions: pd.DataFrame, model_name:str) -> pd.DataFrame:
    # Start filling function
    avg = latest_predictions["predictions"].mean()
    timestamp =  latest_predictions["predictions_time"][0]
    monitoring_df = pd.DataFrame([ [avg,timestamp]], columns=['predictions','predictions_time'])

    with mlflow.start_run():
        metrics = {"predictions":avg}
        mlflow.log_metrics(metrics)
        mlflow.log_params({"predictions_time":timestamp, "model_name": model_name})
    # End filling function
    return monitoring_df
