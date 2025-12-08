from dataclasses import dataclass
from datetime import timedelta, date, datetime
from json import load as load_json
from os import getenv
from typing import Any

import finnhub as fh
import joblib
import pandas as pd
import streamlit as st
import streamlit_float as st_f
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pandas import DataFrame
from plotly import express as px

################
# Initial load
################
st_f.float_init()

@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return kmeans, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_cluster(answers: dict, kmeans, scaler) -> int:
    features = scaler.feature_names_in_
    means = scaler.mean_
    
    
    user_vector = dict(zip(features, means))
    

    for feature, value in answers.items():
        if feature in user_vector:
            user_vector[feature] = float(value)
            
    #  Convert to DataFrame and Scale
    df = pd.DataFrame([user_vector], columns=features)
    df_scaled = scaler.transform(df)
    
    #  Predict
    return int(kmeans.predict(df_scaled)[0])

try:
    if not load_dotenv():
        raise RuntimeError()
except Exception as e:
    st.error("Failed to load environment variables.")
    st.stop()

def reset_session_state():
    st.session_state.clear()

    init_dict = dict(
        stage="survey",
        user_profile={},
        stocks=[],
        chat_history=[],
    )

    for key, value in init_dict.items():
        st.session_state[key] = value

if st.session_state.get("stage") is None:
    reset_session_state()
    st.set_page_config(layout="wide")


################
# Survey stage
################
def make_survey_question(question_name, answers) -> st.radio:
    return st.radio(
        question_name,
        answers,
        horizontal=True,
        width="stretch",
    )


def render_survey_stage():
    st.title("Investor Profiling")
    st.write("Rate your agreement (1-5) to match with an investment persona.")

    # Load models here (cached)
    kmeans, scaler = load_models()

    if kmeans is None or scaler is None:
        st.stop()

    try:
        with open("questions.json", "r") as f:
            survey_questions = load_json(f)
    except FileNotFoundError:
        st.error("Error: questions.json file not found.")
        st.stop()

    with st.form("survey_form"):
        responses = {}
        
        
        for col_name, question in survey_questions.items():
                    st.write(question) 
                    responses[col_name] = st.radio(
                        label=question, 
                        options=[1, 2, 3, 4, 5],
                        horizontal=True, 
                        index=2, 
                        label_visibility="collapsed" 
                    )
                    st.write("---")

        submitted = st.form_submit_button("Analyze Profile")

        if submitted:
            # Save specific responses (for the AI Chat later)
            st.session_state["user_profile"]["responses"] = responses
            
            
            predicted_id = predict_cluster(responses, kmeans, scaler)
            st.session_state["user_profile"]["cluster_id"] = predicted_id
            
            
            st.session_state["stage"] = "processing"
            st.rerun()


################
# Processing recommendations stage
################
@dataclass
class StockInfo:
    name: str
    ticker_symbol: str
    price: float | None
    price_to_earnings: float | None
    history: DataFrame | None
    description: str
    news: list[dict[str, Any]] | None = None
    dividend_yield: float | None = None
    volume: int | None = None

def get_llm_provider_env_dict(provider: str) -> dict[str, str]:
    provider = provider.upper()
    return dict(
        base_url=getenv(f"{provider}_ENDPOINT"),
        api_key=getenv(f"{provider}_API_KEY"),
        model=getenv(f"{provider}_MODEL"),
    )

@st.cache_data(ttl=300)
def get_finnhub_news(ticker: str) -> list[dict[str, Any]]:
    fh_client = fh.Client(api_key=getenv("FINNHUB_API_KEY"))

    today = date.today().isoformat()
    past = (date.today() - timedelta(days=7)).isoformat()

    return fh_client.company_news(ticker, _from=past, to=today)

def get_stock_info(tickers: list[str]) -> list[StockInfo]:
    stocks = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_name = stock.info.get("shortName", None)
            if stock_name is None:
                raise ValueError(f"No stock found for {ticker}.")

            hist = stock.history("5y").rename(columns={"Close": "Price"})
            
            current_price = stock.info.get("currentPrice")
            if current_price is None:
                current_price = stock.info.get("regularMarketPrice")
            if current_price is None:
                current_price = stock.info.get("navPrice")

            # Final check
            if (current_price is None or str(current_price) == "nan") and not hist.empty:
                last_valid = hist["Price"].dropna().iloc[-1]
                current_price = float(last_valid)


            stocks.append(StockInfo(
                name = stock_name,
                ticker_symbol = ticker,
                price = current_price,
                price_to_earnings = stock.info.get("trailingPE", None),
                dividend_yield = stock.info.get("dividendYield", None),
                volume = stock.info.get("volume") or stock.info.get("regularMarketVolume"),
                history = hist[["Price"]] if len(hist) > 0 else None,
                description = stock.info.get(
                    "longBusinessSummary",
                    "No description available.",
                ).replace("$", "\$"),
                news = get_finnhub_news(ticker),
            ))

        except Exception as e:
            continue

    if len(stocks) == 0:
        st.error("Unable to fetch stock data.")
        st.stop()

    return stocks

def render_processing_stage():

    llm = ChatOpenAI(
        **get_llm_provider_env_dict(getenv("DEFAULT_LLM_PROVIDER")),
        temperature=0.6,
    )

    st.title("Generating recommendations...")
    progress_bar = st.progress(0)


    
    progress_bar.progress(25, "Mapping responses...")

    with open("clusters.json", "r") as clusters_file:
        clusters = load_json(clusters_file)

    # getting cluster
    c_id = st.session_state["user_profile"].get("cluster_id", 0)
    if 0 <= c_id < len(clusters):
        st.session_state["user_profile"]["cluster_result"] = clusters[c_id]
    else:
        st.error(f"Error: Calculated Cluster ID {c_id} is out of range.")
        st.stop()

    #Get AI recommendation
    progress_bar.progress(50, "Consulting recommender...")

    cr = st.session_state.get("user_profile").get("cluster_result")

    specific_answers = st.session_state["user_profile"].get("responses", {})

    # We should consider loading this from another file.
    st.session_state.get("chat_history").append(SystemMessage(content=f"""
        You are an AI agent meant to recommend tickers to check out for users
        based on clustering profile from a short survey.
        
        This user's result was:
        - Profile: {cr["profile"]}
        - User Specific Answers (1-5 scale): {specific_answers} 
        - Why: {cr["why"]}
        - Investment Profile: {cr["investment_profile"]}
        - Priority: {cr["priority"]}
        - Examples: {", ".join(cr["examples"])}
        
        Recommend exactly 3 unique stock tickers that align with this profile
        description. Return only the stock tickers separated by a space. Format
        these for yFinance. Do not add anything else.
        
        Example response: NVDA BRK-B AAPL
    """))

    try:
        response = llm.invoke(st.session_state.get("chat_history"))

    # Errors will be handled specifically if there is time later.
    except Exception as e:
        st.error("Unable to connect to AI agent.")
        st.stop()


    # Step 3 - Parse ticker symbols
    progress_bar.progress(75, "Processing recommendations...")

    tickers = list(set(response.content.strip().split()))
    if len(tickers) < 1:
        st.error("No stock tickers found.")
        st.stop()


    # Step 4 - Fetch ticker data
    progress_bar.progress(100, "Fetching market data...")

    st.session_state["stocks"] = get_stock_info(tickers)

    st.session_state["stage"] = "recommendation"
    st.rerun()


################
# results and AI chat stage
################
def prepare_chat():
    if not st.session_state.get("is_first_chat_render", True):
        return None

    st.session_state["is_first_chat_render"] = False

    st.session_state.get("chat_history").append(SystemMessage(
        # Removed line-width limitation.
        # I had to alter this for the current commit.
        # Next commit will move this to its own file.
        content=f"""
            Following that last message, you recommended to the user the following stocks: 
            {", ".join(s.name for s in st.session_state.get("stocks"))}
            
            Follow these behavioral guidelines as closely as possible:
            
            - Continue to assist the user as you are able. If you don't have access to some information, let the user know.
            - If the user doesn't want to speak, just let them know they can speak to you when they wish.
            - Don't recommend any more stocks. If the user wants new recommendations, instruct them to refresh the application.
            - Keep answers concise so you can get rapid feedback from the user.
            - Try to keep things amicable.
            - Don't try to get fancy with formatting. The renderer is very simple.
            - When not writing LaTeX, escape dollar signs with a double backslash.
            - When writing LaTeX, delimit it with dollar signs. Use no other method for LaTeX delimiting. eg: $$ax^{{2}} + bx + c = 0$$ for all LaTeX.
            - Try to avoid nesting formatting such as lists and line breaks inside tables.
            - Make sure your responses are conversational in nature, unless the user explicitly requests otherwise.
            - Finally, please avoid saying something which might offend the user. The descriptions of the profile clusters aren't flattering.
            
            Additional Information:
            
            - The chat window is fairly small (about 400px width by 500px height).
        """,
    ))

    st.session_state.get("chat_history").append(AIMessage(
        content="I'm here to help! Do you have any questions about my "
                "recommendations?",
    ))
    return None

def calculate_roi(history_df, investment_amount=10000, start_year="2020"):
    
    try:
        start_date = f"{start_year}-01-01"
        mask = history_df.index >= start_date
        df_filtered = history_df.loc[mask]
        
        if df_filtered.empty:
            return None, None

        start_price = df_filtered.iloc[0]["Price"]
        current_price = df_filtered.iloc[-1]["Price"]
        
        shares = investment_amount / start_price
        current_value = shares * current_price
        return current_value, start_price
    except Exception:
        return None, None

def render_finnhub_news_cards(company_news: list[dict[str, Any]]):
    # attributes = ["datetime", "source", "headline", "summary", "url"]
    for article in company_news:

        # Unpack into local.
        # `or None` turns empty strings into None.
        headline = article.get("headline", None) or None
        if headline is None:
            continue

        timestamp = article.get("datetime", None)
        source = article.get("source", None) or None
        summary = article.get("summary", None) or None
        url = article.get("url", None) or None

        headline = headline.replace("$", "\$")

        if timestamp is not None:
            date_str = datetime\
                .fromtimestamp(float(timestamp))\
                .strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = None

        with st.container(
            border=True,
            width=320,
            height=240,
            gap=None
        ):
            with st.container(border=False, gap=None):
                if url is not None:
                    st.markdown(f"##### [{headline}]({url})")
                else:
                    st.markdown(f"##### {headline}")

                source_date = " @ ".join(
                    data for data in (source, date_str) if data is not None
                ).replace("$", "\$")

                if source_date != "":
                    st.caption(source_date)

            with st.container(border=False, gap=None):
                if summary is not None:
                    st.markdown(summary.replace("$", "\$"))


def render_stock(stock: StockInfo):
    with st.container(border=True):
        st.subheader(f"{stock.name} ({stock.ticker_symbol})")
        col_chart, col_data = st.columns([0.7, 0.3], gap="medium")
        
        with col_chart:
            if stock.history is not None:
                st.plotly_chart(
                    px.line(stock.history, y="Price"), 
                    use_container_width=True
                )
            else:
                st.warning("No history data available.")


        with col_data:
            st.markdown("Market Data")
            
            st.metric("Current Price", f"${stock.price:,.2f}" if stock.price else "N/A")
            if stock.price_to_earnings:
                # if it is a regular company (AAPL)
                st.metric("P/E Ratio", f"{stock.price_to_earnings:.2f}", help="Price divided by Earnings. Lower values suggest the stock is cheap (Value Investing). Higher values imply high growth.")
            
            elif stock.dividend_yield:
                # If does not have p/e shows dividend yeld
                if stock.dividend_yield < 1:
                                    yield_percent = stock.dividend_yield * 100
                else:
                    yield_percent = stock.dividend_yield    
                st.metric("Dividend Yield", f"{yield_percent:.2f}%", help="Annual return from dividends.")
            
            elif stock.volume:
                # crypto or gold
                vol_str = f"{stock.volume:,.0f}"
                # Formating billions or millions to fit in the screen
                if stock.volume > 1_000_000_000:
                    vol_str = f"{stock.volume/1_000_000_000:.1f}B"
                elif stock.volume > 1_000_000:
                    vol_str = f"{stock.volume/1_000_000:.1f}M"
                
                st.metric("24h Volume", vol_str, help="Trading volume. High volume = High liquidity/interest.")
            
            else:
                st.metric("Valuation", "N/A")

            
            st.divider() 
            
            st.markdown(" If you invested in 2020: ")
            
            if stock.history is not None:
                final_val, start_price = calculate_roi(stock.history)
                
                if final_val:

                    delta = f"{((final_val - 10000) / 10000) * 100:.1f}%"
                    
                    st.metric(
                        label="Your $10k would be:", 
                        value=f"${final_val:,.2f}", 
                        delta=delta
                    )
                    st.caption(f"Entry Price (2020): ${start_price:.2f}")
                else:
                    st.info("Not enough data since 2020.")

        with st.expander("Business Summary"):
            st.caption(stock.description)
        with st.expander("📰 Latest News", expanded=False):
            if stock.news:
                with st.container(
                        border=False,
                        horizontal=True,
                        horizontal_alignment="center",
                        gap="small",
                        height=700
                ):
                    render_finnhub_news_cards(stock.news)
            else:
                st.info("No recent news found.")

def render_chat():

    # Inject CSS
    st.markdown(
        """
        <style>
        /* Open and close button */
        [data-testid="stPopover"] {
            position: fixed;
            bottom: 25px;
            right: 25px;
            width: auto;
            z-index: 9999;
        }
        
        /* Popover window */
        [data-baseweb="popover"] {
            width: 40vw;
            z-index: 9999;
            padding: 5px;
            opacity: 0.975;
        }
        
        /* Chat messages */
        [data-testid="stChatMessage"] {
            padding: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    prepare_chat()
    chat_hist = st.session_state.get("chat_history")

    popover = st.popover(label="AI Assistant", icon=":material/smart_toy:", type="secondary")
    with popover:
        messages_container = st.container(height=450, border=False, gap=None)

        with messages_container:
            for message in chat_hist:
                match message:
                    case SystemMessage():
                        pass
                    case AIMessage():
                        st.chat_message("assistant").write(message.content)
                    case HumanMessage():
                        st.chat_message("user").write(message.content)

        if user_message := st.chat_input("⌨️"):

            user_message = user_message.replace("$", "\$")
            chat_hist.append(HumanMessage(content=user_message))

            with messages_container:
                st.chat_message("user").write(user_message.replace("$", "\$"))

            search_tool = TavilySearchResults(max_results=2)
            tools = [search_tool]

            llm = ChatOpenAI(
                **get_llm_provider_env_dict(getenv("DEFAULT_LLM_PROVIDER")),
                temperature=0.6,
            )

            agent_app = create_agent(model=llm, tools=tools)

            with messages_container:
                with st.chat_message("assistant"):
                    # This container is needed to suppress a ghosting bug in
                    # Streamlit.
                    with st.container():
                        with st.spinner("", show_time=True):
                            response = agent_app.invoke({"messages": chat_hist})

                            final_answer = response["messages"][-1].content
                            st.write(final_answer)

                            chat_hist.append(AIMessage(content=final_answer))


def render_start_over_button():
    css = """
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        width: auto;
    """

    container = st.container()
    with container:
        if st.button("Start Over", on_click=reset_session_state):
            st.rerun()

    container.float(css)


def render_recommendation_stage():

    st.title("Your Recommendations")

    with st.container(gap="large"):
        for stock in st.session_state.get("stocks"):
            render_stock(stock)

    render_start_over_button()
    render_chat()


################
# Session stage mapping
################
if st.session_state["stage"] == "survey":
    render_survey_stage()

elif st.session_state["stage"] == "processing":
    render_processing_stage()

elif st.session_state["stage"] == "recommendation":
    render_recommendation_stage()