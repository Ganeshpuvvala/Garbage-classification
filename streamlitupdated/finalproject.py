# 📦 Step 1: Install dependencies
!pip install -q streamlit pyngrok tensorflow pillow matplotlib

# 📁 Step 2: Build directories
!mkdir -p Garbage-classification/model
!mkdir -p Garbage-classification/pages
!mkdir -p Garbage-classification/utils

# 🧠 Step 3: Create and train a simple dummy model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

X = np.random.rand(100, 224, 224, 3).astype("float32")
y = to_categorical(np.random.randint(0, 6, 100), 6)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=10, verbose=1)
model.save("Garbage-classification/model/model.h5")

# 📝 Step 4: Create Streamlit files
# app.py
with open("Garbage-classification/app.py", "w") as f:
    f.write("""
import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Garbage Classifier", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Metrics"])

if page == "Home":
    st.title("Welcome to Garbage Classification App")
    st.write(\"\"\"
        This application allows you to classify garbage images into different categories.

        **Pages:**
        - Predict: Upload or capture image to classify.
        - Metrics: View model performance metrics.
    \"\"\")

elif page == "Predict":
    from pages.page_predict import predict_page
    predict_page()

elif page == "Metrics":
    from pages.page_metrics import metrics_page
    metrics_page()
""")

# page_predict.py
with open("Garbage-classification/pages/page_predict.py", "w") as f:
    f.write("""
def predict_page():
    import streamlit as st
    from PIL import Image
    from utils.model_utils import load_model, predict_image

    st.header("📸 Garbage Image Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        model = load_model()
        label, confidence = predict_image(image, model)
        st.success(f"Predicted: {label} with {confidence:.2f}% confidence")
""")

# page_metrics.py
with open("Garbage-classification/pages/page_metrics.py", "w") as f:
    f.write("""
def metrics_page():
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    st.header("📊 Model Performance Metrics")
    st.write("Sample Confusion Matrix (Placeholder)")

    data = np.array([[50, 2, 3], [4, 45, 1], [2, 3, 48]])
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='Blues')
    fig.colorbar(cax)
    st.pyplot(fig)
""")

# model_utils.py
with open("Garbage-classification/utils/model_utils.py", "w") as f:
    f.write("""
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("Garbage-classification/model/model.h5")

def predict_image(image, model):
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array

    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    label_index = np.argmax(predictions)
    confidence = predictions[label_index] * 100
    class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    return class_names[label_index], confidence
""")

# ✅ Step 5: Launch the app with ngrok
from pyngrok import ngrok
ngrok.set_auth_token("ngrok_auth_token")
import threading

def run():
    !streamlit run Garbage-classification/app.py --server.headless true --server.port 8501

public_url = ngrok.connect(8501)
print("🔗 Open your app:", public_url)
threading.Thread(target=run).start()
