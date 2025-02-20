import onnx
import onnxruntime as rt
#program to check if onnx model was converted properly
# Load the ONNX model
model_path = "isolation_forest_model.onnx"
onnx_model = onnx.load(model_path)

model = rt.InferenceSession(model_path)
