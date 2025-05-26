from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def build_pipeline():
    """
    Returns an sklearn Pipeline:
     - Imputes missing values
     - Scales features
     - Trains XGBClassifier
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("classifier", XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])
