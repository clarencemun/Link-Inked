# Link-Inked

![App screenshot](screenshot.jpg)

Link-Inked is a Streamlit-based application that leverages Google News RSS feeds to display relevant news articles. Users can specify search criteria based on topics, locations, and custom search queries. The application also integrates with the Ollama LLM to identify the most compelling headlines and generate professional, LinkedIn-style comments for each selected headline.

## Features

- **Dynamic RSS Feed URL Generation**: Users can dynamically generate URLs to fetch news based on various criteria such as top headlines, specific topics, or custom search parameters.
- **LLM Integration for Headline Selection**: Utilizes the Ollama LLM to identify and highlight the top headlines that are most likely to engage readers.
- **Automated Comment Generation**: Generates insightful comments suitable for professional social media platforms like LinkedIn, enhancing user engagement and discussion.
- **Responsive User Interface**: Built with Streamlit, the UI is intuitive and responsive, making it accessible across different devices and screen sizes.

## Installation

To set up and run Link-Inked locally, you will need Python and pip installed on your machine.
You will also need Ollama with Llama3:8b running in the background for this project.

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/clarencemun/Link-Inked.git
   cd Link-Inked
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch the application by running the following command in your terminal:

```bash
streamlit run Link-Inked.py
```

Access the application in your web browser at `http://localhost:8501`.

## Contributing

Contributions to Link-Inked are welcome! For major changes, please open an issue first to discuss what you would like to change. Ensure any pull requests are made against predetermined branch policies.