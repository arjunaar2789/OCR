import easyocr
from PIL import Image
import streamlit as st
import warnings
import numpy as np
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit app title
st.title('Optical Character Recognition')
st.markdown('## Extract text from images')
st.markdown('')

# List of available languages in EasyOCR
languages = {
    'Abaza': 'abq',
    'Adyghe': 'ady',
    'Afrikaans': 'af',
    'Angika': 'ang',
    'Arabic': 'ar',
    'Assamese': 'as',
    'Azerbaijani': 'az',
    'Belarusian': 'be',
    'Bulgarian': 'bg',
    'Bengali': 'bn',
    'Tibetan': 'bo',
    'Bosnian': 'bs',
    'Catalan': 'ca',
    'Czech': 'cs',
    'Welsh': 'cy',
    'Danish': 'da',
    'German': 'de',
    'Dzongkha': 'dz',
    'Greek': 'el',
    'English': 'en',
    'Spanish': 'es',
    'Estonian': 'et',
    'Persian': 'fa',
    'French': 'fr',
    'Irish': 'ga',
    'Goan Konkani': 'gom',
    'Hindi': 'hi',
    'Croatian': 'hr',
    'Hungarian': 'hu',
    'Armenian': 'hy',
    'Indonesian': 'id',
    'Icelandic': 'is',
    'Italian': 'it',
    'Japanese': 'ja',
    'Javanese': 'jv',
    'Georgian': 'ka',
    'Kazakh': 'kk',
    'Central Khmer': 'km',
    'Korean': 'ko',
    'Kurdish': 'ku',
    'Kyrgyz': 'ky',
    'Luxembourgish': 'lb',
    'Ganda': 'lg',
    'Lingala': 'ln',
    'Lao': 'lo',
    'Lithuanian': 'lt',
    'Latvian': 'lv',
    'Maithili': 'mai',
    'Maori': 'mi',
    'Macedonian': 'mk',
    'Malayalam': 'ml',
    'Mongolian': 'mn',
    'Marathi': 'mr',
    'Malay': 'ms',
    'Maltese': 'mt',
    'Burmese': 'my',
    'Dutch': 'nl',
    'Norwegian': 'no',
    'Occitan': 'oc',
    'Panjabi': 'pa',
    'Polish': 'pl',
    'Pashto': 'ps',
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Sanskrit': 'sa',
    'Sinhala': 'si',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Albanian': 'sq',
    'Serbian': 'sr',
    'Swedish': 'sv',
    'Swahili': 'sw',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Thai': 'th',
    'Tagalog': 'tl',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Uzbek': 'uz',
    'Vietnamese': 'vi',
    'Chinese (Simplified)': 'ch_sim',
    'Chinese (Traditional)': 'ch_tra'
}

# Dropdown for language selection with full names
language_name = st.selectbox(
    'Select Language',
    list(languages.keys())
)

# Get the corresponding language code
language_code = languages[language_name]

# File uploader for image
image = st.file_uploader('Upload the image', type=['jpg', 'jpeg', 'png'])

@st.cache_resource
def load_model(language_code):
    """Load the EasyOCR model for the selected language."""
    reader = easyocr.Reader([language_code], gpu=False)
    return reader

def get_text_download_link(text, filename):
    """Generate a link to download the extracted text."""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Extracted Text</a>'
    return href

if image is not None:
    try:
        # Display uploaded image
        input_img = Image.open(image)
        st.image(input_img, caption='Uploaded Image')

        # Load OCR model
        reader = load_model(language_code)

        # Process image with OCR
        with st.spinner('Processing...'):
            result = reader.readtext(np.array(input_img))
            st.success('Processing Complete')

        # Display the results
        st.markdown('### Extracted Text')
        result_text = []
        for bbox, text, prob in result:
            st.write(f'{text} (Confidence: {prob:.2f})')
            result_text.append(text)

        # Provide download link for extracted text
        download_link = get_text_download_link("\n".join(result_text), "extracted_text.txt")
        st.markdown(download_link, unsafe_allow_html=True)

        # Save results to history
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "image": input_img,
            "text": result_text
        })
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write('Please upload an image')

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
