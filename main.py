import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dataclasses import dataclass
import yfinance as yf
from pandas import DataFrame
from plotly import express as px
from random import sample
from json import load as load_json


################
# Initial load
################
st.set_page_config(layout="wide")

if "stage" not in st.session_state:
    st.session_state["stage"] = "survey"

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {}



def reset_session_state():
    st.session_state["stage"] = "survey"
    st.session_state["user_profile"] = {}


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

def render_survey():
    st.title("Survey")
    st.write("Take this survey, to help us narrow down recommendations.")

    with st.form("survey_form"):
        # We should consider loading these from a file.
        q0 = make_survey_question(
            "Question 0",
            [1,2,3,4,5],
        )
        q1 = make_survey_question(
            "Question 1",
            [1,2,3,4,5],
        )
        q2 = make_survey_question(
            "Question 2",
            [1,2,3,4,5],
        )
        q3 = make_survey_question(
            "Question 3",
            [1,2,3,4,5],
        )

        submitted_survey = st.form_submit_button("Continue")

        if submitted_survey:
            st.session_state["user_profile"] = {
                "q0": q0,
                "q1": q1,
                "q2": q2,
                "q3": q3,
            }

            st.session_state["stage"] = "recommending"
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

def get_stock_info(tickers: list[str]) -> list[StockInfo]:
    stocks = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_name = stock.info.get("shortName", None)
            if stock_name is None:
                raise ValueError(f"No stock found for {ticker}.")

            hist = stock.history("1y").rename(columns={"Close": "Price"})

            stocks.append(StockInfo(
                name = stock_name,
                ticker_symbol = ticker,
                price = stock.info.get("currentPrice", None),
                price_to_earnings = stock.info.get("trailingPE", None),
                history = hist[["Price"]] if len(hist) > 0 else None,
                description = stock.info.get(
                    "longBusinessSummary",
                    "No description available.",
                ),
            ))

        except Exception as e:
            pass

    if len(stocks) == 0:
        st.error("Unable to fetch stock data.")
        st.stop()

    return stocks


def render_recommending_phase():

    # CHANGE TO EXTERNAL ENDPOINT AND ENV FILE KEYS WHEN READY.
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="LM-Studio",
        temperature=0.5,
    )

    st.title("Generating recommendations...")
    progress_bar = st.progress(0)


    # Step 1 - Determine cluster
    progress_bar.progress(25, "Mapping responses...")

    with open("clusters.json", "r") as clusters_file:
        clusters = load_json(clusters_file)

    # REPLACE WITH CLUSTER MAPPING WHEN READY.
    st.session_state["cluster_result"] = sample(clusters, 1)[0]

    # Step 2 - Get AI recommendation
    progress_bar.progress(50, "Consulting recommender...")

    cr = st.session_state.get("cluster_result", "Unknown")
    # We should consider loading this from another file.
    prompt = f"""
    You are an AI agent meant to recommend tickers to check out for users based
    on clustering profile from a short survey.
    
    This user's result was:
    - Profile: {cr["profile"]}
    - Why: {cr["why"]}
    - Investment Profile: {cr["investment_profile"]}
    - Priority: {cr["priority"]}
    - Examples: {", ".join(cr["examples"])}
    
    Recommend exactly 3 unique stock tickers that align with this profile description.
    Return only the stock tickers separated by a space. Format these for 
    yFinance. Do not add anything else.
    
    Example response: NVDA BRK-B AAPL
    """

    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
        ])
    # Errors will be handled specifically if there is time later.
    except Exception as e:
        st.error("Unable to connect to AI agent.")
        st.stop()


    # Step 3 - Parse ticker symbols
    progress_bar.progress(75, "Processing recommendations...")

    st.session_state["tickers"] = list(set(response.content.strip().split()))
    if len(st.session_state["tickers"]) < 1:
        st.error("No stock tickers found.")


    # Step 4 - Fetch ticker data
    progress_bar.progress(100, "Fetching market data...")

    st.session_state["stocks"] = get_stock_info(st.session_state["tickers"])

    st.session_state["stage"] = "chat"
    st.rerun()


################
# results and AI chat stage
################
def render_agent_chat():

    st.title("Your Recommendations")
    st.write(st.session_state["cluster_result"])

    for stock in st.session_state["stocks"]:
        st.divider()
        st.subheader(stock.name)

        ticker_col, price_col, pte_col = st.columns(3)
        if stock.ticker_symbol is not None:
            ticker_col.markdown(
                f"""
                <div style='text-align: left;'>
                    Symbol: {stock.ticker_symbol}
                </div>
                """,
                unsafe_allow_html=True,
            )
        if stock.price is not None:
            price_col.markdown(
                f"""
                <div style='text-align: center;'>
                    Current Price: {stock.price}
                </div>
                """,
                unsafe_allow_html=True,
            )
        if stock.price_to_earnings is not None:
            pte_col.markdown(
                f"""
                <div style='text-align: right;'>
                    Price to Earnings: {stock.price_to_earnings}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if stock.history is not None:
            st.plotly_chart(px.line(stock.history, y="Price"))

        with st.expander("Business Summary"):
            st.write(stock.description)

    st.divider()
    st.subheader("AI Agent")
    st.chat_message("assistant").write(
        "This is placeholder text. In the future, I'll be able to have a full "
        "conversation with you!")

    if prompt := st.chat_input("Converse with the AI agent..."):
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write("Sorry, but I'm not yet an AI agent!")

    st.button("Start Over", on_click=reset_session_state)


################
# Session stage mapping
################
if st.session_state['stage'] == 'survey':
    render_survey()

elif st.session_state['stage'] == 'recommending':
    render_recommending_phase()

elif st.session_state['stage'] == 'chat':
    render_agent_chat()