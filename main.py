import cv2
import numpy as np
import streamlit as st
from keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
  model = MobileNetV2(
      weights='imagenet'
  )
  return model

def preprocess_image(img):
  img = np.array(img)
  img = cv2.resize(img, (224, 224))
  img = preprocess_input(img)
  img = np.expand_dims(img, axis=0)
  return img

def classify_image(model, image):
  try:
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top = 3)[0] #This takes the top 3 predictions and [0] means it takes the first response given by the model
    return decoded_predictions
  except Exception as e:
    st.error(f'Error classifying image: {str(e)}')
    return None
  
def main():
  st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="centered"
  )
  st.title("AI Image Classifier")
  st.write("Upload an image and let AI tell you what is in the image.")

  @st.cache_resource #This allows the model to be cached so that it doesn't have to be loaded every time the page is refreshed
  def load_cached_model():
    return load_model()
  
  model = load_cached_model()

  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = st.image(
      uploaded_file,
      caption="Uploaded Image",
      width='stretch'
    )
    btn = st.button("Classify Image")
    if btn:
      with st.spinner("Classifying image..."):
        image = Image.open(uploaded_file)
        result = classify_image(model, image)
        if result:
          st.subheader("Classification Results")
          for _, label, score in result:
            st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
  main()

