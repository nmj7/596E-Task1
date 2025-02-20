import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Define categories for each feature separately
categories = [
    ["MerchantA", "MerchantB", "MerchantC", "MerchantD", "MerchantE",
     "MerchantF", "MerchantG", "MerchantH", "MerchantI", "MerchantJ"],  # Merchants

    ["Purchase", "Transfer", "Withdrawal"],  # Transaction types

    ["New York", "London", "Tokyo", "Los Angeles", "San Francisco"]  # Locations
]

# Convert each category list into a separate column (not rows)
categories_reshaped = [[m, t, l] for m in categories[0] for t in categories[1] for l in categories[2]]

# Convert to NumPy array
categories_np = np.array(categories_reshaped, dtype=object)

# Train the OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(categories_np)  # Now the data is in the correct shape

# Save the trained encoder to encoder.pkl
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Encoder trained and saved as 'encoder.pkl' successfully!")
