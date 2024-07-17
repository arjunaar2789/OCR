# File: app.py
import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
import base64

st.title('EASY OCR - Extract Text from Images')
st.markdown('## Optical Character Recognition')
st.markdown('')

# Language selection
language = st.selectbox(
    "Select OCR language",
    ["en", "fr", "es", "de", "it", "pt", "zh", "ja", "ko", "ru"]
)

image = st.file_uploader(label='Upload your image here', type=['png', 'jpg', 'jpeg'])

@st.cache_resource
def load_model(language):
    reader = ocr.Reader([language], gpu=False)
    return reader

if image is not None:
    input_image = Image.open(image)
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    reader = load_model(language)

    with st.spinner('Processing...'):
        result = reader.readtext(np.array(input_image))
        result_text = [text[1] for text in result]

        st.success('Processing complete!')

        st.markdown('### Extracted Text')
        for line in result_text:
            st.write(line)
        
        st.balloons()

        # Save results to history
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "image": input_image,
            "text": result_text
        })

        # Provide download link for extracted text
        def get_text_download_link(text, filename):
            b64 = base64.b64encode(text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Extracted Text</a>'
            return href

        download_link = get_text_download_link("\n".join(result_text), "extracted_text.txt")
        st.markdown(download_link, unsafe_allow_html=True)

else:
    st.write('Please upload an image.')

# Display history of uploaded images and extracted text
if 'history' in st.session_state and st.session_state.history:
    st.markdown('## History')
    for idx, record in enumerate(st.session_state.history):
        st.markdown(f"### Image {idx + 1}")
        st.image(record["image"], use_column_width=True)
        st.markdown('Extracted Text:')
        for line in record["text"]:
            st.write(line)
st.caption('Made by Arjun A A R')
