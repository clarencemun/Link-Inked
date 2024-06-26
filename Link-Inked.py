import urllib.parse
import streamlit as st
import feedparser
import ollama
from PIL import Image
import os
import uuid  # Import uuid for generating unique identifiers
import streamlit.components.v1 as components

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the banner image using a relative path
image_path = os.path.join(BASE_DIR, "banner.png")
image = Image.open(image_path)

# Display the banner image
st.image(image, width=200)

costar_prompt = """
# CONTEXT #
A business analyst with a strong interest and knowledge in data science and AI needs to generate a reserved, professional, and insightful comment for a LinkedIn post.

#########

# OBJECTIVE #
Create a LinkedIn comment that is reserved, professional, insightful, and avoids the use of exclamation marks. The comment should be no more than 200 words, quotes one key point from the article, contextualises that quote, as well as include a sentence from the first person perspective that demonstrates the analyst's domain knowledge.

#########

# STYLE #
The comment should be succinct, professional, and insightful.

#########

# TONE #
The tone should be reserved and professional.

#########

# AUDIENCE #
The intended audience is the LinkedIn network of the business analyst, including peers, potential employers, and industry professionals.

#########

# RESPONSE #
Print only the LinkedIn comment and nothing but the LinkedIn comment in text format.

#############

# START ANALYSIS #

[ARTICLE]
"""

# Function to use Ollama to pick the top headlines
def pick_top_headlines(headlines, n=3):
    numbered_headlines = [f"{i + 1}. {title}" for i, (title, _) in enumerate(headlines)]
    input_text = " and ".join(numbered_headlines)
    input_text = f"As a business analyst with a strong interest and knowledge in data science and AI, pick the top {n} most interesting headlines from the following, and provide the serial number along with the headline: {input_text}"

    # Define the conversation history format
    conversation_history = [
        {'role': 'user', 'content': input_text}
    ]

    # Use Ollama LLM to analyze the headlines and pick the top ones
    stream = ollama.chat(
        model='llama3:8b',
        messages=conversation_history,
        stream=True
    )

    # Collect response and parse the selected headlines
    response = ""
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            response += chunk['message']['content']

    selected_headlines_with_numbers = []
    for line in response.split("\n"):
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


# Function to generate comments for LinkedIn
def generate_comment(headline):
    # This prompt setup should match the expected input structure for Ollama
    conversation_history = [
        {'role': 'user', 'content': f"'{costar_prompt}' '{headline}'."}
    ]

    # Assuming 'ollama.chat()' expects 'messages' and not 'prompt'
    response = ollama.chat(
        model='llama3:8b',
        messages=conversation_history,
        stream=True  # Adjust this according to the actual function definition if 'stream' is not needed
    )

    # Collect response from the stream
    comment = ""
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            comment += chunk['message']['content']

    return comment.strip()


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
    navigator.clipboard.writeText(textToCopy)
        }}
        </script>
    """
    components.html(html_content, height=30)  # Adjust height as necessary


# Streamlit UI setup
feed_type = st.selectbox('Select News Type', ['Top Headlines', 'By Topic', 'By Country', 'By Search Terms'], index=0)

# Conditional input fields based on feed type
search_terms = []
default_terms = ['Artificial Intelligence', 'Data Science', 'Business', 'Data Analytics', '']  # Default search terms
if feed_type == 'By Search Terms':
    st.write("Enter up to five search terms:")
    for i in range(5):
        term = st.text_input(f'Search Term {i+1}', value=default_terms[i])
        search_terms.append(term)


topic = st.selectbox('Select Topic', ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH'], index=0) if feed_type == 'By Topic' else ''
location = st.text_input('Enter Location', '') if feed_type == 'By Country' else ''
time_frame = st.text_input('Enter Time Frame (e.g., 12h, 1d, 7d, 3m)', '1d') if feed_type == 'By Search Terms' else ''
language = "en-SG"
country = "SG"


# Generate RSS URL and fetch news
if st.button('Generate'):
    # Assuming `generate_rss_url` needs parameters like feed_type, search_terms, topic, location, time_frame, language, and country
    rss_url = generate_rss_url(feed_type, search_terms, topic, location, time_frame, language, country)
    headlines = fetch_news_from_rss(rss_url)
    if headlines:
        top_headlines = pick_top_headlines(headlines, 3)
        for title, link in top_headlines:
            comment = generate_comment(title)
            # Append the URL to the comment
            full_comment = f"{comment}\n\nRead more here: {link}"
            unique_id = str(uuid.uuid4())
            st.subheader(title)
            st.write(full_comment)
            copy_button(full_comment, unique_id)
            st.write('---')
    else:
        st.write("No news items found.")