import os
import re
import uuid
import urllib.parse
import streamlit as st
import feedparser
from newspaper import Article, Config
from googlenewsdecoder import gnewsdecoder
from PIL import Image
import streamlit.components.v1 as components
import google.generativeai as genai
import ollama
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Set base directory and load banner image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "banner.png")

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, width=200)
else:
    st.write("Banner image not found.")

# Set model temperature
model_temperature = 0.1

# Set model max tokens
model_max_tokens = 300

# Configure API keys
GEMINI_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)
gemini_model_name = 'gemini-1.5-pro'

# Gemini API interaction function
def generate_gemini_comment(user_prompt, model_name=gemini_model_name):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=model_temperature,
                max_output_tokens=model_max_tokens,
            )
        )
        return response.text
    except Exception as e:
        st.error(f"An error occurred with Gemini: {e}")
        return ""

# DeepSeek API interaction function
def generate_deepseek_comment(user_prompt, model_name='DeepSeek-R1'):
    endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
    key = os.getenv("AZURE_INFERENCE_SDK_KEY")
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    messages = [
        UserMessage(content=user_prompt)
    ]

    response = client.complete(
        messages=messages,
        model=model_name,
        max_tokens=model_max_tokens,
        temperature=model_temperature,
    )

    response_text = response.choices[0].message.content
    return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

# Streamlit sidebar for model selection
with st.sidebar:
    st.header("Settings")
    model_type = st.radio("Select Model Type", ('Cloud', 'Local'), key='model_type')

    if model_type == 'Local':
        ollama_model = st.selectbox(
            'Select your Ollama model',
            ['gemma3', 'gemma3:27b', 'llama3.2:3b', 'llama3.3', 'mistral', 'mistral-small:24b', 'qwen2.5:14b', 'qwen2.5:32b', 'deepseek-llm:67b', 'deepseek-r1:32b'],
            index=0,
            key='ollama_model'
        )
    else:
        cloud_model = st.selectbox(
            'Select Cloud Model',
            ('GPT-4o', 'Gemini 1.5 Pro', 'DeepSeek-R1'),
            key='cloud_model'
        )

# Azure OpenAI setup
if model_type == 'Cloud' and cloud_model == 'GPT-4o':
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
    client = AzureOpenAI(
        azure_endpoint=st.secrets["AZURE_ENDPOINT"],
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=st.secrets["AZURE_API_VERSION"]
    )

# Prompt for generating LinkedIn comments
costar_prompt = """
# CONTEXT #
A business analyst and Gen AI consultant with a strong interest and knowledge in data science and AI needs to generate a reserved, professional, and insightful comment for a LinkedIn article. If the article is not related to technology, provide a knowledgeable and insightful comment that demonstrates strong domain knowledge. The article content will be provided at the end of the prompt. As it is webscraped, it may contain HTML tags, ads, links to other articles and other formatting issues. do your best to focus on the content of the article and ignore the rest.

#########

# OBJECTIVE #
Create a LinkedIn comment that is reserved, professional, insightful, and avoids the use of exclamation marks or bullet points. Be detailed but focused. Do not address the author directly, and cut unnecessary pleasantries. If the article is tech-related, talk about the underlying technologies and implications where applicable. If the article is not tech-related, provide a knowledgeable and insightful comment that provides contextually relevant insights. Subtly include philosophical, ethical, or societal perspectives that add value to the discussion.

#########

# STYLE #
The comment should be engaging, succinct yet detailed, professional, and insightful. Use simple language while providing a deeper demonstration of your domain knowledge, whether in technology or another field.

#########

# TONE #
The tone should be reserved and professional.

#########

# AUDIENCE #
The intended audience is the LinkedIn network of the business analyst cum Gen AI practitioner, including peers, potential employers, and industry professionals.

#########

# RESPONSE #
Print only the LinkedIn comment and nothing but the LinkedIn comment in text format.
The comment must not be more than 150 words. Include a brief summary of the article, highlighting key points. Include 1 sentence from the first person perspective that demonstrates the expert's domain knowledge.

#############

# START ANALYSIS #
This is the end of the prompt, and the start of the article content
"""

# Helper function to remove <think> tags
def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Function to generate comments using Ollama
def generate_comment(article_content):
    prompt = f"{costar_prompt}\n{article_content}"
    conversation_history = [{'role': 'user', 'content': prompt}]

    try:
        response = ollama.chat(
            model=ollama_model,
            messages=conversation_history,
            options={'temperature': model_temperature, 'max_tokens': model_max_tokens},
            stream=True
        )
        response_text = "".join(chunk['message']['content'] for chunk in response if 'message' in chunk and 'content' in chunk['message'])
        return remove_think_tags(response_text)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Function to generate comments using Azure OpenAI
def generate_azure_comment(article_content):
    prompt = f"{costar_prompt}\n[ARTICLE]\n{article_content}\n"
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=model_temperature,
            max_tokens=model_max_tokens
        )
        return remove_think_tags(response.choices[0].message.content)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Decode Google News URL
def decode_google_news_url_wrapper(url):
    try:
        decoded_url_response = gnewsdecoder(url)
        if decoded_url_response.get("status"):
            return decoded_url_response["decoded_url"]
    except Exception:
        # Silently handle the error and return the original URL
        return url

# Extract article content using Newspaper with User-Agent
def extract_article_content(url):
    try:
        # Use a more recent and commonly used user-agent string
        user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        config = Config()
        config.browser_user_agent = user_agent

        # Decode the URL if it is a Google News URL
        actual_url = decode_google_news_url_wrapper(url)
        if not actual_url:
            actual_url = url  # Use the original URL if decoding fails

        article = Article(actual_url, config=config)
        article.download()
        article.parse()

        title = article.title
        article_content = article.text

        return f"{title}. {article_content.strip()}".strip(), actual_url
    except Exception as e:
        st.error(f"An error occurred while extracting the article: {e}")
        return "", url

# Utility functions for RSS feeds and article extraction
def fetch_news_from_rss(url):
    feed = feedparser.parse(url)
    return [(entry.title, entry.link) for entry in feed.entries]

def generate_rss_url(feed_type, search_terms='', topic='', location='', time_frame='12h', language='en-SG', country='SG'):
    base_url = "https://news.google.com/rss"
    if feed_type == 'Top Headlines':
        return f"{base_url}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Topic':
        return f"{base_url}/headlines/section/topic/{urllib.parse.quote_plus(topic.upper())}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Country':
        return f"{base_url}/headlines/section/geo/{urllib.parse.quote_plus(location)}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Search Terms':
        formatted_query = "%20OR%20".join([urllib.parse.quote_plus(term.strip()) for term in search_terms if term])
        return f"{base_url}/search?q={formatted_query}+when:{time_frame}&hl={language}&gl={country}&ceid={country}:{language}"
    return base_url

def copy_button(comment, unique_id, link=None):
    additional_text = f"\n\n\nRead more here:\n{link}" if link else ""
    full_text_to_copy = comment + additional_text
    html_content = f"""
        <textarea id='textarea-{unique_id}' style='opacity: 0; position: absolute; z-index: -1; left: -9999px;'>
{full_text_to_copy}
        </textarea>
        <button onclick="copyToClipboard('{unique_id}')">Copy</button>
        <script>
        function copyToClipboard(unique_id) {{
            const copyText = document.getElementById('textarea-' + unique_id);
            const textToCopy = copyText.value.trim();
            navigator.clipboard.writeText(textToCopy);
        }}
        </script>
    """
    components.html(html_content, height=30)

# Streamlit UI setup
feed_type = st.selectbox('Select News Type', ['By Search Terms', 'Generate from URL', 'Manual Input', 'Top Headlines', 'By Topic', 'By Country'], index=0, key='feed_type')

# Conditional input fields based on feed type
if feed_type == 'Generate from URL':
    article_url = st.text_input("Enter the URL:", key='article_url_generate')
    if st.button('Generate', key='generate_button'):
        article_content, actual_url = extract_article_content(article_url) if article_url.strip() else ('', article_url)
        if article_content:
            if model_type == 'Cloud' and cloud_model == 'Gemini 1.5 Pro':
                comment = generate_gemini_comment(article_content)
            elif model_type == 'Cloud' and cloud_model == 'DeepSeek-R1':
                comment = generate_deepseek_comment(article_content)
            elif model_type == 'Local':
                comment = generate_comment(article_content)
            else:
                comment = generate_azure_comment(article_content)
            unique_id = str(uuid.uuid4())
            st.subheader("Generated Comment:")
            st.write(comment)
            st.write(f"\nRead more here:\n{actual_url}")
            copy_button(comment, unique_id, link=actual_url)
elif feed_type == 'Manual Input':
    st.header('Generate LinkedIn Comment Manually')
    article_content = st.text_area("Paste the article content here:", key='manual_article_content')
    article_url = st.text_input("Enter the Article URL (optional):", key='manual_article_url')
    if st.button('Generate Comment', key='manual_generate_button'):
        if article_content.strip():
            if model_type == 'Cloud' and cloud_model == 'Gemini 1.5 Pro':
                comment = generate_gemini_comment(article_content)
            elif model_type == 'Cloud' and cloud_model == 'DeepSeek-R1':
                comment = generate_deepseek_comment(article_content)
            elif model_type == 'Local':
                comment = generate_comment(article_content)
            else:
                comment = generate_azure_comment(article_content)
            unique_id = str(uuid.uuid4())
            st.subheader("Generated Comment:")
            st.write(comment)
            if article_url.strip():
                st.write(f"\nRead more here:\n{article_url}")
            copy_button(comment, unique_id, link=article_url)
        else:
            st.write("Please paste the article content to generate a comment.")
else:
    search_terms = []
    default_terms = ['Artificial Intelligence', 'AI', 'Data Science', 'Business', 'Data Analytics', 'Machine Learning', 'LLM', 'NLP', '', '']
    if feed_type == 'By Search Terms':
        st.write("Enter up to ten search terms:")
        for i in range(10):
            term = st.text_input(f'Search Term {i+1}', value=default_terms[i], key=f'search_term_{i}')
            search_terms.append(term)

    topic = st.selectbox('Select Topic', ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH'], index=0, key='topic') if feed_type == 'By Topic' else ''
    location = st.text_input('Enter Location', 'Singapore', key='location') if feed_type == 'By Country' else ''
    time_frame = st.text_input('Enter Time Frame (e.g., 12h, 1d, 7d, 3m)', '12h', key='time_frame') if feed_type == 'By Search Terms' else ''

    if st.button('Generate', key='generate_headlines_button'):
        rss_url = generate_rss_url(feed_type, search_terms, topic, location, time_frame)
        headlines = fetch_news_from_rss(rss_url)
        if headlines:
            for title, link in headlines[:7]:
                article_content, actual_url = extract_article_content(link)
                if article_content:
                    st.subheader(title)
                    if model_type == 'Cloud' and cloud_model == 'Gemini 1.5 Pro':
                        comment = generate_gemini_comment(article_content)
                    elif model_type == 'Cloud' and cloud_model == 'DeepSeek-R1':
                        comment = generate_deepseek_comment(article_content)
                    elif model_type == 'Local':
                        comment = generate_comment(article_content)
                    else:
                        comment = generate_azure_comment(article_content)
                    unique_id = str(uuid.uuid4())
                    st.write(comment)
                    st.write(f"\nRead more here:\n{actual_url}")
                    copy_button(comment, unique_id, link=actual_url)
                    st.write('---')

# Streamlit UI setup for improving an existing comment
st.header('Improve an Existing Comment')
existing_comment = st.text_area("Paste the existing comment here:", key='existing_comment')
improvement_prompt = st.text_area("Enter instructions for improving the comment:", key='improvement_prompt')
if st.button('Improve Comment', key='improve_button'):
    if existing_comment.strip() and improvement_prompt.strip():
        url_match = re.search(r'(https?://\S+)', existing_comment)
        extracted_url = url_match.group(0) if url_match else None

        improve_prompt = f"""
# CONTEXT #
A business analyst and Gen AI practitioner with a strong interest and knowledge in data science and AI needs to improve an existing LinkedIn comment based on the additional instructions provided. If the article is not related to technology, the business analyst should adopt the persona of an expert in that specific topic.

#########

# OBJECTIVE #
Improve the existing LinkedIn comment while maintaining its reserved, professional, and insightful tone. Avoid the use of exclamation marks and bullet points and keep the comment under 150 words. The improved comment should be enhanced with the given instructions.

#########

# STYLE #
The comment should be engaging, succinct, professional, and insightful. Provide a more nuanced demonstration of domain knowledge in that field.

#########

# TONE #
The tone should be reserved and professional.

#########

# RESPONSE #
Print only the improved LinkedIn comment and nothing but the improved LinkedIn comment in text format.

#############

# COMMENT #
{existing_comment}

# INSTRUCTIONS #
{improvement_prompt}
"""
        try:
            if model_type == 'Cloud' and cloud_model == 'Gemini 1.5 Pro':
                improved_comment = generate_gemini_comment(improve_prompt)
            elif model_type == 'Cloud' and cloud_model == 'DeepSeek-R1':
                improved_comment = generate_deepseek_comment(improve_prompt)
            elif model_type == 'Local':
                response = ollama.chat(
                    model=ollama_model,
                    messages=[{'role': 'user', 'content': improve_prompt}],
                    options={'temperature': model_temperature, 'max_tokens': model_max_tokens},
                    stream=True
                )
                improved_comment = "".join(chunk['message']['content'] for chunk in response if 'message' in chunk and 'content' in chunk['message'])
                improved_comment = remove_think_tags(improved_comment.strip())
            else:
                response = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[{'role': 'user', 'content': improve_prompt}],
                    temperature=model_temperature,
                    max_tokens=model_max_tokens
                )
                improved_comment = remove_think_tags(response.choices[0].message.content.strip())
            unique_id = str(uuid.uuid4())
            st.subheader("Improved Comment:")
            st.write(improved_comment)
            if extracted_url:
                st.write(f"\nRead more here: \n{extracted_url}")
            full_text_to_copy = f"{improved_comment}\n\n\nRead more here:\n{extracted_url if extracted_url else 'N/A'}"
            html_content = f"""
                <textarea id='textarea-{unique_id}' style='opacity: 0; position: absolute; z-index: -1; left: -9999px;'>
{full_text_to_copy}
                </textarea>
                <button onclick="copyToClipboard('{unique_id}')">Copy</button>
                <script>
                function copyToClipboard(unique_id) {{
                    const copyText = document.getElementById('textarea-' + unique_id);
                    const textToCopy = copyText.value.trim();
                    navigator.clipboard.writeText(textToCopy);
                }}
                </script>
            """
            components.html(html_content, height=30)
        except Exception as e:
            st.error(f"An error occurred while improving the comment: {e}")
    else:
        st.warning("Please provide both the existing comment and improvement instructions.")