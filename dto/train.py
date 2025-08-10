from typing import List
from pydantic import BaseModel

# Define a Pydantic model for the request body
class Train(BaseModel):
    trainDataPercentage: float
    featureMethod: str  # PCA, LDA, ICA, CNN
    classifierMethods: List[str] | None = None  # SVM, KNN, NB, RF, DT
    path: str | None = None
    category: str | None = None