import numpy as np
import pickle
import streamlit as st

# load model
loaded_model = pickle.load(open("trained_model.sav", "rb"))


# Prediction function

def breast_cancer_prediction(input_data):
    input_data = (
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56,
        0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288,
        0.2977,
        0.07259)

    # change to numpy array
    input_data_np_array = np.asarray(input_data)

    # reshape numpy array as we are predicting one value (diagnosis)
    input_data_reshaped = input_data_np_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "Malignant"

    else:
        return "Benign"


def main():
    # Title
    st.title("Breast Cancer Prediction Web App")

    # Input data
    radius_mean = st.text_input("Input Radius Mean")
    texture_mean = st.text_input("Input Texture Mean")
    perimeter_mean = st.text_input("Input Perimeter Mean")
    area_mean = st.text_input("Input Area Mean")
    smoothness_mean = st.text_input("Input Smoothness Mean")
    compactness_mean = st.text_input("Input Compactness Mean")
    concavity_mean = st.text_input("Input Concavity Mean")
    concave_points_mean = st.text_input("Input Concave Points Mean")
    symmetry_mean = st.text_input("Input Symmetry Mean")
    fractal_dimension_mean = st.text_input("Input Fractal Dimension Mean")
    radius_se = st.text_input("Input Radius se")
    texture_se = st.text_input("Input Texture se")
    perimeter_se = st.text_input("Input Perimeter se")
    area_se = st.text_input("Input Area se")
    smoothness_se = st.text_input("Input Smoothness se")
    compactness_se = st.text_input("Input Compactness se")
    concavity_se = st.text_input("Input Concavity se")
    concave_points_se = st.text_input("Input Concave Points se")
    symmetry_se = st.text_input("Input Symmetry se")
    fractal_dimension_se = st.text_input("Input Fractal Dimension se")
    radius_worst = st.text_input("Input Radius Worst")
    texture_worst = st.text_input("Input Texture Worst")
    perimeter_worst = st.text_input("Input Perimeter Worst")
    area_worst = st.text_input("Input Area Worst")
    smoothness_worst = st.text_input("Input Smoothness Worst")
    compactness_worst = st.text_input("Input Compactness Worst")
    concavity_worst = st.text_input("Input Concavity Worst")
    concave_points_worst = st.text_input("Input Concave Points Worst")
    symmetry_worst = st.text_input("Input Symmetry Worst")
    fractal_dimension_worst = st.text_input("Input Fractal Dimension Worst")

    # Prediction
    diagnosis = ""

    # Predict Button
    if st.button("Breast Cancer Test Result"):
        diagnosis = breast_cancer_prediction(
            [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
             concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
             smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
             radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
             concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst])

        st.success(diagnosis)


if __name__ == "__main__":
    main()
