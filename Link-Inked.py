import urllib.parse
import streamlit as st
import feedparser
from PIL import Image
import os
import uuid
import streamlit.components.v1 as components
import openai
from openai import AzureOpenAI



# Set up Azure OpenAI API key and endpoint
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]

# Initialise Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_ENDPOINT"],
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Ensure API key is stored securely in environment variables
    api_version=st.secrets["AZURE_API_VERSION"]
)

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the banner image using a relative path
image_path = os.path.join(BASE_DIR, "banner.png")
if os.path.exists(image_path):
    image = Image.open(image_path)
    # Display the banner image
    st.image(image, width=200)
else:
    st.write("Banner image not found.")

costar_prompt = """
# CONTEXT #
A business analyst and Gen AI practitioner with a strong interest and knowledge in data science and AI needs to generate a reserved, professional, and insightful comment for a LinkedIn article.

#########

# OBJECTIVE #
Create a LinkedIn comment that is reserved, professional, insightful, and avoids the use of exclamation marks. Talk about the underlying technologies where applicable. The comment should be between 250 and 300 words and include a summary of the article and a sentence from the first person perspective that demonstrates the analyst's domain knowledge.

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
"""

# Function to generate comments for LinkedIn using your client object
def generate_comment(article_content):
    prompt = f"{costar_prompt}\n[ARTICLE]\n{article_content}\n"

    # Call the GPT model using the client object and handle response correctly
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        return response_text.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Function to generate comments for LinkedIn manually using article content and URL
def generate_manual_comment(article_content, article_url):
    prompt = f"{costar_prompt}\n[ARTICLE]\n{article_content}\n"

    # Call the GPT model using the client object and handle response correctly
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        comment = response_text.strip()
        if article_url.strip():
            comment += f"\n\nRead the article here:\n\n{article_url}"
        return comment
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Function to pick the top headlines using your client object
def pick_top_headlines(headlines, n=5):
    numbered_headlines = [f"{i + 1}. {title}" for i, (title, _) in enumerate(headlines)]
    input_text = "\n".join(numbered_headlines)
    input_text = f"As a business analyst and Gen AI practitioner with a strong interest and knowledge in data science and AI, pick the top {n} most interesting headlines from the following list, and provide the serial number along with the headline:\n{input_text}"

    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{'role': 'user', 'content': input_text}],
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

    selected_headlines_with_numbers = []
    for line in reply.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            try:
                number, headline = line.split(". ", 1)
                selected_headlines_with_numbers.append((int(number), headline))
            except ValueError:
                pass

    # Return only the selected headlines based on their numbers
    selected_indices = [number - 1 for number, headline in selected_headlines_with_numbers]
    return [headlines[index] for index in selected_indices if index < len(headlines)]

def fetch_news_from_rss(url):
    feed = feedparser.parse(url)
    return [(entry.title, entry.link) for entry in feed.entries]

def generate_rss_url(feed_type, search_terms='', topic='', location='', time_frame='1d', language='en-SG', country='SG'):
    base_url = "https://news.google.com/rss"
    if feed_type == 'Top Headlines':
        return f"{base_url}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Topic':
        return f"{base_url}/headlines/section/topic/{urllib.parse.quote_plus(topic.upper())}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Country':
        return f"{base_url}/headlines/section/geo/{urllib.parse.quote_plus(location)}?hl={language}&gl={country}&ceid={country}:{language}"
    elif feed_type == 'By Search Terms':
        # Join the search terms with 'OR' and encode them
        formatted_query = "%20OR%20".join([urllib.parse.quote_plus(term.strip()) for term in search_terms if term])
        return f"{base_url}/search?q={formatted_query}+when:{time_frame}&hl={language}&gl={country}&ceid={country}:{language}"
    return base_url  # default to top headlines if no type matches

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
            navigator.clipboard.writeText(textToCopy);
        }}
        </script>
    """
    components.html(html_content, height=30)  # Adjust height as necessary

# Streamlit UI setup
feed_type = st.selectbox('Select News Type', ['Top Headlines', 'By Topic', 'By Country', 'By Search Terms', 'Manual Input'], index=0)

# Conditional input fields based on feed type
search_terms = []
default_terms = ['Artificial Intelligence', 'AI', 'Data Science', 'Business', 'Data Analytics', 'Machine Learning', 'LLM', 'NLP', '', '']  # Default search terms
if feed_type == 'By Search Terms':
    st.write("Enter up to ten search terms:")
    for i in range(10):
        term = st.text_input(f'Search Term {i+1}', value=default_terms[i])
        search_terms.append(term)

topic = st.selectbox('Select Topic', ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH'], index=0) if feed_type == 'By Topic' else ''
location = st.text_input('Enter Location', 'Singapore') if feed_type == 'By Country' else ''
time_frame = st.text_input('Enter Time Frame (e.g., 12h, 1d, 7d, 3m)', '1d') if feed_type == 'By Search Terms' else ''
language = "en-SG"
country = "SG"

# Generate RSS URL and fetch news
if feed_type != 'Manual Input' and st.button('Generate'):
    # Generate RSS URL based on user input
    rss_url = generate_rss_url(feed_type, search_terms, topic, location, time_frame, language, country)
    headlines = fetch_news_from_rss(rss_url)
    if headlines:
        top_headlines = pick_top_headlines(headlines, 5)
        if top_headlines:
            for title, link in top_headlines:
                st.subheader(title)
                comment = generate_comment(title)
                if comment:
                    # Append the URL to the comment
                    full_comment = f"{comment}\n\nRead more here: {link}"
                    unique_id = str(uuid.uuid4())
                    st.write(full_comment)
                    copy_button(full_comment, unique_id)
                    st.write('---')
                else:
                    st.write("Failed to generate comment.")
        else:
            st.write("Failed to select top headlines.")
    else:
        st.write("No news items found.")

# Streamlit UI setup for manual comment generation
if feed_type == 'Manual Input':
    st.header('Generate LinkedIn Comment Manually')

    # Input fields for article URL and content
    article_url = st.text_input("Paste the article URL here:")
    article_content = st.text_area("Paste the article content here:")

    if st.button('Generate Comment'):
        if article_content.strip():
            comment = generate_manual_comment(article_content, article_url)
            unique_id = str(uuid.uuid4())
            st.subheader("Generated Comment:")
            st.write(comment)
            copy_button(comment, unique_id)
        else:
            st.write("Please paste the article content to generate a comment.")
