import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_page_config(
    page_title="Potato Disease Classification"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('potato_model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Disease Classification
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_reshape = (cv2.resize(img, dsize=(256, 256),    interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    string = "Prediction : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.success(string)
    else:
        st.warning(string)
html_link = """
    Made by <a href="https://vanshhhhh.github.io/" style="color:green;" target="_blank">Vansh Sharma</a>
    """
st.markdown(html_link, unsafe_allow_html=True)