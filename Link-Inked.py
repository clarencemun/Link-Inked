import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import ollama
import random

# Setup ChromeDriver
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # for running headlessly
    driver = webdriver.Chrome(options=options)
    return driver

def fetch_headlines(urls):
    driver = setup_driver()
    valid_headlines = []
    try:
        for url in urls:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            articles_attempt = 0
            max_attempts = 3  # Maximum attempts to fetch articles if they become stale

            while articles_attempt < max_attempts:
                try:
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.IFHyqb.DeXSAc")))
                    articles = driver.find_elements(By.CSS_SELECTOR, "article.IFHyqb.DeXSAc")
                    
                    for article in articles[:15]:
                        try:
                            headline_tag = article.find_element(By.CSS_SELECTOR, "a.JtKRv")
                            link_tag = article.find_element(By.CSS_SELECTOR, "a.WwrzSb")
                            source_tag = article.find_element(By.CSS_SELECTOR, "div.vr1PYe")

                            headline = headline_tag.text.strip()
                            link = link_tag.get_attribute("href")
                            source = source_tag.text.strip()

                            if source:
                                valid_headlines.append((headline, link))
                        except StaleElementReferenceException:
                            print("Stale element within article loop, retrying...")
                            break  # Break the inner loop to retry the whole article fetch
                    else:
                        # If the inner loop did not break (no stale elements), exit while loop
                        break
                except StaleElementReferenceException:
                    print("Stale element encountered, retrying fetch...")
                    articles_attempt += 1

                if articles_attempt >= max_attempts:
                    raise Exception("Failed to fetch articles after multiple attempts due to stale elements.")

    finally:
        driver.quit()

    return valid_headlines


def pick_top_headlines(headlines, n=15):
    numbered_headlines = [f"{i + 1}. {headline}" for i, (headline, link) in enumerate(headlines)]
    input_text = " and ".join(numbered_headlines)
    input_text = f"Pick the top {n} most interesting headlines from the following, and provide the serial number along with the headline: {input_text}"

    conversation_history = [
        {'role': 'system', 'content': 'You are a business analyst who is equally learned about AI and Data Science.'},
        {'role': 'user', 'content': input_text}
    ]

    stream = ollama.chat(
        model='llama3:8b',
        messages=conversation_history,
        stream=True
    )

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

    selected_indices = [number - 1 for number, headline in selected_headlines_with_numbers]
    return [headlines[index] for index in selected_indices]

def fetch_article_content(url):
    driver = setup_driver()
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        paragraphs = []
        attempt = 0
        max_attempts = 3  # Set a maximum number of attempts to avoid infinite loops
        
        while attempt < max_attempts and not paragraphs:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "p")))
                paragraph_elements = driver.find_elements(By.CSS_SELECTOR, "p")
                paragraphs = [p.text for p in paragraph_elements if p.text]
            except StaleElementReferenceException:
                print("Encountered a stale element, retrying...")
                attempt += 1
                if attempt >= max_attempts:
                    raise Exception("Failed to fetch content after multiple attempts due to stale elements.")
            except TimeoutException:
                print("Timeout waiting for article content to load.")
                break
        
        return "\n".join(paragraphs)
    finally:
        driver.quit()


def generate_comment(headline, article_content):
    conversation_history = [
        {'role': 'system', 'content': 'You are a business analyst who is equally learned about AI and Data Science.'},
        {'role': 'user', 'content': f"Generate a reserved, professional, and insightful comment from a third-person perspective, avoiding the use of exclamation marks and of not more than 100 words for LinkedIn for the article titled '{headline}'. The content is: {article_content}"}
    ]

    stream = ollama.chat(
        model='llama3:8b',
        messages=conversation_history,
        stream=True
    )

    response = ""
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            response += chunk['message']['content']
    return response

# Default URLs to fetch news articles
default_urls = [
    "https://news.google.com/topics/CAAqKAgKIiJDQkFTRXdvS0wyMHZNR3AwTTE5eE14SUZaVzR0UjBJb0FBUAE?hl=en-SG&gl=SG&ceid=SG%3Aen",
    "https://news.google.com/topics/CAAqJAgKIh5DQkFTRUFvSEwyMHZNRzFyZWhJRlpXNHRSMElvQUFQAQ?hl=en-SG&gl=SG&ceid=SG%3Aen",
    "https://news.google.com/topics/CAAqKAgKIiJDQkFTRXdvS0wyMHZNREp4TTJNMU5oSUZaVzR0UjBJb0FBUAE?hl=en-SG&gl=SG&ceid=SG%3Aen",
    "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSkwyMHZNR00yZW10MEVnVmxiaTFIUWlnQVAB?hl=en-SG&gl=SG&ceid=SG%3Aen",
    "https://news.google.com/topics/CAAqKQgKIiNDQkFTRkFvTEwyY3ZNVEl4YW01eE1XMFNCV1Z1TFVkQ0tBQVAB?hl=en-SG&gl=SG&ceid=SG%3Aen",
    "https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNRFZtYkdZU0JXVnVMVWRDS0FBUAE?hl=en-SG&gl=SG&ceid=SG%3Aen"
]

# Streamlit app setup
st.title("Link-Inked")

urls_input = st.text_area("Enter URLs (one per line):", "\n".join(default_urls))
urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

if st.button("Generate Post"):
    st.subheader("Generating post...")
    valid_headlines = fetch_headlines(urls)
    top_headlines = pick_top_headlines(valid_headlines, 15)
    for index, (headline, url) in enumerate(top_headlines):
        article_content = fetch_article_content(url)
        comment = generate_comment(headline, article_content)
        comment += f"\n\nRead more here: {url}"
        st.write(f"**{headline}**")
        st.write(comment)
