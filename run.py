import onnx
import onnxruntime as rt

# Load the ONNX model
model_path = "isolation_forest_model.onnx"
onnx_model = onnx.load(model_path)

model = rt.InferenceSession(model_path)
