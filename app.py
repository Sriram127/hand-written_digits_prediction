import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from src.digit_model import load_or_train_model, predict_digit


st.set_page_config(page_title="Handwritten Digit Prediction", page_icon="🔢", layout="wide")

st.title("Handwritten Digit Prediction")
st.caption("Upload a handwritten digit image and classify it with a lightweight scikit-learn model.")

model, metrics = load_or_train_model()

prediction_tab, metrics_tab, about_tab = st.tabs(["Prediction", "Model Metrics", "About"])

with prediction_tab:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction, top_predictions = predict_digit(image)

        image_col, result_col = st.columns([1, 2])
        with image_col:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with result_col:
            st.metric("Predicted digit", prediction)
            st.write("Top predictions")
            for item in top_predictions:
                st.progress(
                    item["confidence"],
                    text=f"Digit {item['digit']} - {item['confidence']:.1%}",
                )
    else:
        st.info("Upload a clear image of one digit on a light background.")

with metrics_tab:
    st.metric("Validation accuracy", f"{metrics['accuracy']:.2%}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with about_tab:
    st.write(
        "This demo uses the built-in scikit-learn digits dataset. Images are converted "
        "to grayscale, resized to 8x8 pixels, normalized, and passed to a Random Forest classifier."
    )
    st.code("streamlit run app.py", language="bash")
