from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path= "C:\\Users\\This  PC\\Documents\\project mlops\\Data\\heart.csv")


#mlflow ui --backend-store-uri "file:C:\\Users\\This  PC\\AppData\\Roaming\\zenml\\local_stores\\5123b831-5882-46a0-942a-d978d3a380db\\mlruns