from typing import TypedDict
import numpy as np
import pandas as pd
import pickle
import joblib
import onnxruntime as rt
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    ResponseBody, BatchTextResponse, TextResponse, TaskSchema, InputSchema,
    ParameterSchema, InputType, BatchTextInput, TextInput
)
from flask_ml.flask_ml_server.models import RangedFloatParameterDescriptor, FloatRangeDescriptor

# Load OneHotEncoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

if not hasattr(encoder, "transform"):
    raise ValueError("Loaded encoder is not a valid OneHotEncoder transformer.")

# Initialize Flask-ML server
ml_server = MLServer(__name__)

# Define input format
class AnomalyDetectionInputs(TypedDict):
    amount: BatchTextInput
    merchant: BatchTextInput
    transaction_type: BatchTextInput
    location: BatchTextInput

# Define parameters format
class AnomalyDetectionParameters(TypedDict):
    threshold: float

# Task schema function
def create_detect_anomalies_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(key="amount", input_type=InputType.BATCHTEXT, label="Transaction amount"),
            InputSchema(key="merchant", input_type=InputType.BATCHTEXT, label="Merchant name"),
            InputSchema(key="transaction_type", input_type=InputType.BATCHTEXT, label="Type of transaction"),
            InputSchema(key="location", input_type=InputType.BATCHTEXT, label="Location of the transaction"),
        ],
        parameters=[
            ParameterSchema(
                key="threshold",
                label="Threshold for anomaly detection",
                value=RangedFloatParameterDescriptor(
                    default=0.5,
                    range=FloatRangeDescriptor(min=0.0, max=1.0)
                ),
            ),
        ],
    )

@ml_server.route("/detect_anomalies", task_schema_func=create_detect_anomalies_task_schema)
def detect_anomalies(inputs: AnomalyDetectionInputs, parameters: AnomalyDetectionParameters):
    """
    Detect anomalies in financial transactions.
    """
    try:
        # Extract text values
        amount = float(inputs["amount"].texts[0].text)
        merchant = inputs["merchant"].texts[0].text
        transaction_type = inputs["transaction_type"].texts[0].text
        location = inputs["location"].texts[0].text

        print(f"Received Input: {amount}, {merchant}, {transaction_type}, {location}")

        # Preprocess input
        input_data = preprocess_input(amount, merchant, transaction_type, location, encoder)

        # Run ONNX inference
        prediction = predict_anomalies(input_data)

        if len(prediction) == 0:
            raise ValueError("ONNX model returned an empty prediction.")

        result_text = "Anomaly detected" if prediction[0] == -1 else "Transaction is normal"

        return ResponseBody(root=BatchTextResponse(texts=[TextResponse(value=result_text)]))

    except Exception as e:
        return ResponseBody(root=BatchTextResponse(texts=[TextResponse(value=f"Error: {str(e)}")]))


def preprocess_input(amount, merchant, transaction_type, location, encoder):
    """
    Preprocess input data for anomaly detection.
    """
    try:
        # DataFrame for encoding (ensure correct column names)
        categorical_df = pd.DataFrame([[merchant, transaction_type, location]], 
                                      columns=["Merchant", "TransactionType", "Location"])  

        print(f"Data Before Encoding:\n{categorical_df}")

        # One-hot encoding
        encoded_cols = encoder.transform(categorical_df)

        # Convert sparse to dense if necessary
        if hasattr(encoded_cols, "toarray"):
            encoded_cols = encoded_cols.toarray()

        print(f"Encoded Data Shape: {encoded_cols.shape}")

        # Combine `amount` with encoded features
        final_input = np.hstack(([amount], encoded_cols.ravel())).astype(np.float32)

        print(f"Final Input Shape: {final_input.shape}")

        return final_input.reshape(1, -1)  # Ensure correct shape

    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")


def predict_anomalies(input_data):
    """
    Run ONNX model to predict anomalies.
    """
    try:
        sess = rt.InferenceSession("isolation_forest_model.onnx")

        # Get model input name
        input_name = sess.get_inputs()[0].name  
        expected_features = sess.get_inputs()[0].shape[1]  

        # Check input shape
        if input_data.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} input features, got {input_data.shape[1]}.")

        # Run ONNX model
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: input_data.astype(np.float32)})[0]

        return pred_onx

    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([])  # Return empty array on error


# Run Flask-ML server
if __name__ == "__main__":
    ml_server.run()
