# Link-Inked

![App screenshot](screenshot.jpg)

## Description
Link-Inked is a web application built using Streamlit and Selenium, designed to automate the process of fetching and displaying news headlines from specified URLs. The application allows users to select the most interesting headlines, fetch the content of the articles, and generate insightful comments suitable for professional networks like LinkedIn.

## Features
- Fetch news headlines from user-defined URLs.
- Interactive selection of top headlines.
- Fetch full articles and generate professional comments for LinkedIn posts.

## Installation

### Requirements
- Python 3.6 or newer
- Streamlit
- Selenium
- ChromeDriver (compatible with your Chrome version)

### Setup
1. Clone the repository:
git clone <repository-url>
cd <repository-directory>


2. Install the required Python libraries:
pip install streamlit selenium ollama


3. Ensure ChromeDriver is installed and in your PATH. You can download it from:
[ChromeDriver - WebDriver for Chrome](https://sites.google.com/a/chromium.org/chromedriver/).

## Usage

1. Start the application:
streamlit run Link-Inked.py


2. The application will launch in your default web browser. You can interact with it by entering URLs and selecting options through the Streamlit interface.

3. Enter the URLs from which you want to fetch news headlines in the text area provided.

4. Click on 'Generate Post' to fetch headlines, select top headlines, fetch the full article, and generate a comment.

## Contributing
Contributions to Link-Inked are welcome! Please feel free to fork the repository, make changes, and submit pull requests.