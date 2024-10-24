import streamlit as st
import os
from openai import AzureOpenAI
import uuid  # Import uuid for generating unique identifiers
import streamlit.components.v1 as components
from PIL import Image

# Set up Azure OpenAI API key and endpoint
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the banner image using a relative path
image_path = os.path.join(BASE_DIR, "banner.png")
image = Image.open(image_path)

# Display the banner image
st.image(image, width=200)

# Initialise Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_ENDPOINT"],
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Ensure API key is stored securely in environment variables
    api_version=st.secrets["AZURE_API_VERSION"]
)

costar_prompt = """
# CONTEXT #
A business analyst and Gen AI practitioner with a strong interest and knowledge in data science and AI needs to generate a reserved, professional, and insightful comment for a LinkedIn article.

#########

# OBJECTIVE #
Create a LinkedIn comment that is reserved, professional, insightful, and avoids the use of exclamation marks. Talk about the underlying technologies where applicable. The comment should be between 200 and 250 words and include a summary of the article and a sentence from the first person perspective that demonstrates the analyst's domain knowledge.

#########

# STYLE #
The comment should be engaging, succinct, professional, and insightful. Provide a more nuanced demonstration of your domain knowledge.

#########

# TONE #
The tone should be reserved and professional.

#########

# AUDIENCE #
The intended audience is the LinkedIn network of the business analyst cum Gen AI practitioner, including peers, potential employers, and industry professionals.

#########

# RESPONSE #
Print only the LinkedIn comment and nothing but the LinkedIn comment in text format.

#############

# START ANALYSIS #

[ARTICLE]
"""

# Function to generate comments for LinkedIn
def generate_comment(article_content):
    prompt = f"{costar_prompt} {article_content}"

    # Call the GPT model using the client object and handle response correctly
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7
    )

    response_text = response.choices[0].message.content
    return response_text.strip()

def copy_button(comment, unique_id):
    html_content = f"""
        <textarea id='textarea-{unique_id}' style='opacity: 0; position: absolute; z-index: -1; left: -9999px;'>
            {comment}
        </textarea>
        <button onclick="copyToClipboard('{unique_id}')">Copy</button>
        <script>
        function copyToClipboard(unique_id) {{
            const copyText = document.getElementById('textarea-' + unique_id);
            const textToCopy = copyText.value.trim();  // Use trim() to remove any leading or trailing whitespace
            navigator.clipboard.writeText(textToCopy)
        }}
        </script>
    """
    components.html(html_content, height=30)  # Adjust height as necessary

# Streamlit UI setup
st.header('Generate LinkedIn Comment')

# Input fields for article URL and content
article_url = st.text_input("Paste the article URL here:")
article_content = st.text_area("Paste the article content here:")

if st.button('Generate Comment'):
    if article_content.strip():
        comment = generate_comment(article_content)
        if article_url.strip():
            comment += f"\n\nRead the article here:\n\n{article_url}"
        unique_id = str(uuid.uuid4())
        st.subheader("Generated Comment:")
        st.write(comment)
        copy_button(comment, unique_id)
    else:
        st.write("Please paste the article content to generate a comment.")