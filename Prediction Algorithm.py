import numpy as np
import pickle

# load model
loaded_model = pickle.load(open("trained_model.sav", "rb"))

input_data = (
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56,
    0.008462,
    0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

# change to numpy array
input_data_np_array = np.asarray(input_data)

# reshape numpy array as we are predicting one value (diagnosis)
input_data_reshaped = input_data_np_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("Malignant")

else:
    print("Benign")
