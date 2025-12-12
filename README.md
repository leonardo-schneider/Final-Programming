# Data Analysis of Investment

A personality-based stock recommendation engine that bridges psychometrics with financial data. This application analyzes user personality traits via a condensed Likert survey to recommend stocks, providing financial data, recent news, and an interactive AI agent for a more personalized experience.

## Overview

This project explores the intersection of behavioral psychology and investing. It replaces traditional "risk tolerance" questionnaires with a psychometric approach, mapping fundamental personality traits to market sectors and specific tickers.

**How it works:**

1. **Survey:** User takes a quick 9-item personality assessment.
2. **Recommendation:** Stocks are suggested based on the user's psychographic profile.
3. **Stock View:** Users view 5-year price history, P/E ratios, dividend yields, and business summaries.
4. **Agent Interaction:** Users can chat with an AI agent (powered by LLMs + Tavily Search) to ask follow-up questions about the recommendations.

## Methodology

While the interface is simple, the backend uses a rigorous data reduction pipeline derived from the "Young People Survey" dataset:

1.  **Factor Analysis:** We started with a raw dataset of **135 survey questions**. Using statistical Factor Analysis, we identified the core latent variables, distilling the survey down to **27 key questions**.
2.  **LLM Semantic Mapping:** We utilized a Large Language Model to further condense these 27 questions into just **9 high-impact items**, ensuring the survey remains engaging without sacrificing predictive power.
3.  **Real-Time Data:** The app integrates pulls data from **Yahoo Finance** for stock data, **Finnhub** for weekly news sentiment, and **Tavily** for live web-search capabilities during the chat and recommendation.

## Features

  * **Psychometric Profiling:** A streamlined 9-question Likert scale interface.
  * **Financial Dashboard:** Displays current price, 5-year history charts, P/E ratio, Dividend Yield, and 24hr Volume.
  * **News Integration:** Fetches relevant news articles from the past 7 days via Finnhub.
  * **RAG-Powered Chat:** Discuss portfolio choices with an AI agent that has access to live web search.

## Prerequisites

  * **Python 3.12+**
  * **uv** (Recommended) or anything which can set up a python environment from a requirements.txt file.
  * **API Keys:**
      * **GROQ API Key** (or a local LM Studio server running)
      * **Finnhub API Key** (for stock data & news)
      * **Tavily API Key** (for agent web search)

## Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/leonardo-schneider/Final-Programming.git
    cd Final-Programming
    ```

2.  **Install dependencies:**

    ```bash
    uv sync
    ```

3.  **Configure Environment:**

    Create a `.env` file in the root directory and add your keys:

    ```ini
    GROQ_API_KEY="your_groq_key_here"
    FINNHUB_API_KEY="your_finnhub_key_here"
    TAVILY_API_KEY="your_tavily_key_here"
    ```
    
## Usage

To launch the Streamlit application:

```bash
uv run streamlit run main.py
```

*Note: If you are not using `uv`, ensure your virtual environment is activated and run `streamlit run main.py`.*

## Project Structure

```text
.
├── main.py                         # Application entry point (Streamlit)
├── pyproject.toml                  # Dependency definitions
├── README.md                       # This file
├── requirements.txt                # Dependencies for non-uv managers
├── resources                       # Static assets
│   ├── personality_factors.json
│   ├── prompt_tenets.txt
│   └── survey_traits.json
└── uv.lock                         # Exact dependency versions
```

-----

*This project was created for Programming for Data Science at New College of Florida as an exploration of agentic AI and data reduction techniques.*