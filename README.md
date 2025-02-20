# **FINANCIAL ANOMOLY DETECTION ONNX MODEL**
---
This repo will show how to run this particular model. On passing parameters such as Amount(float), Merchant(string), Transaction_Type(string),and Location(string) in the following format, the model can detect if the transaction is an anomoly or not.

`python -m cli detect_anomalies --amount 100.00 --merchant "MerchantH" --transaction_type "Transfer" --location "London" --threshold 0.5`

The output would be as follows:

`C:\Users\Nikitha John\Documents\596E TASK 1\venv\Lib\site-packages\sklearn\utils\validation.py:2732: UserWarning: X has feature names, but OneHotEncoder was fitted without feature names
  warnings.warn(
Encoded Data Shape: (1, 18)
Final Input Shape: (19,)
        Results:
        Anomaly detected
        Results:
        Anomaly detected`

# **STEPS TO CONVERT IT INTO ONNX MODEL**
---
1. For starters, start a virtual environment(Windows) using
   `python -m venv venv
   venv/bin/activate`
2. Now code out your Financial detection anomoly model as in model.py and then install all your required libraries using the command
   'pip install numpy pandas seaborn matplotlib onnx==1.16.1 onnxruntime skl2onnx pickle scikit-learn' 
3. After making sure model.py runs, in the same model.py file include
   `initial_type = [('float_input', FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model,initial_types=initial_type,target_opset={"ai.onnx.ml": 3, "" : 13} )
onnx_filename = "isolation_forest_model.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Model saved to {onnx_filename}")`
This will convert the py model to onnx format. The onnx file would be called isolation_forest_model.onnx

# **HOW TO CREATE THE SERVER(FLASK-ML) AND THEN THE CLI**
---
This part provides the API to run the onnx model
1. run the following command to install the dependcies required for this part of the program
   `pip install Flask-ML pylanche`
2. After this make the pyrightconfig.json to your project directory. It should contain the following:
   `{
    "python.analysis.typeCheckingMode": "basic"
}`
3. Now make your train_encoder.py file that contains OneHotEncoder for categorical variables that are as follows
categories = [["MerchantA", "MerchantB", "MerchantC", "MerchantD", "MerchantE", "MerchantF", "MerchantG", "MerchantH", "MerchantI", "MerchantJ"], ["Purchase", "Transfer", "Withdrawal"],  ["New York", "London", "Tokyo", "Los Angeles", "San Francisco"] ]
4.Now make your server.py which will contain your Flask-ML server, make sure you load your onnx model and use
`sess = rt.InferenceSession("isolation_forest_model.onnx")`
This initializes an ONNX Runtime session, allowing the model to be used for inference.

#**API Endpoint(in server.py)**
1. Endpoint: /detect_anomalies
2. Method: POST
3. The request and response format will both be in json
   **Request:**
   `{
    "amount": { "float": [{ "float": "120.50" }] },
    "merchant": { "texts": [{ "text": "MerchantH" }] },
    "transaction_type": { "texts": [{ "text": "Transfer" }] },
    "location": { "texts": [{ "text": "New York" }] }
}`
   **Response:**
   `{
    "root": {
        "texts": [{ "value": "Anomaly detected" }]
    }
}`

#** HOW TO RUN THIS PROGRAM VIA CLI**
---
1. Install the follwoing library using pip and make sure you have python 3.12+
   `pip install argparse`
2. Start the server using
   `python server.py`
3. Once thats running, switch to another terminal and run
   `python -m cli detect_anomalies --amount 100.00 --merchant "MerchantH" --transaction_type "Transfer" --location "London" --threshold 0.5`
Replace the parameters as per your choice
4. The `--help` command displays all the availible commands and make sure isolation_forest_model.onnx is in the same directory as cli.py

#**DEMO**
---

*server.py*
![image](https://github.com/user-attachments/assets/587c4888-52b1-47ea-8668-0df7c1bc8d14)

*client.py*
![image](https://github.com/user-attachments/assets/2c6f757b-26e2-4a08-8042-63c2d85d91ca)







