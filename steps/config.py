from zenml.steps import BaseParameters

class ModelConfig(BaseParameters):

    model_name: list= ["CatBoostClassifier", "GradientBoostingClassifier", "RandomForestClassifier"]