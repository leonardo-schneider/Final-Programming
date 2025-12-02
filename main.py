import streamlit as st
import streamlit_float as st_f
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dataclasses import dataclass
import yfinance as yf
from pandas import DataFrame
from plotly import express as px
from random import sample
from json import load as load_json
from dotenv import load_dotenv
from os import getenv
import joblib
import pandas as pd
import numpy as np

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

SURVEY_QUESTIONS = {
    "Shopping": "How much do you enjoy shopping?",
    "Reading": "Do you enjoy reading books?",
    "Romantic": "Do you consider yourself romantic?",
    "Theatre": "Do you enjoy going to the theatre?",
    "PC": "How much time do you spend on the PC?",
    "Cars": "Are you interested in cars?",
    "Life struggles": "Do you often feel emotional or struggle with life?",
    "Politics": "Are you interested in politics?",
    "Economy Management": "Are you interested in economics/management?" 
}

def render_survey_stage():
    st.title("Investor Profiling")
    st.write("Rate your agreement (1-5) to match with an investment persona.")

    # Load models here (cached)
    kmeans, scaler = load_models()

    if kmeans is None or scaler is None:
        st.stop()

    with st.form("survey_form"):
        responses = {}
        
        
        for col_name, question in SURVEY_QUESTIONS.items():
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

def get_llm_provider_env_dict(provider: str) -> dict[str, str]:
    provider = provider.upper()
    return dict(
        base_url=getenv(f"{provider}_ENDPOINT"),
        api_key=getenv(f"{provider}_API_KEY"),
        model=getenv(f"{provider}_MODEL"),
    )

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
        content=f"""
            Following that last message, you recommended to the user the
            following stocks: 
            {", ".join(s.name for s in st.session_state.get("stocks"))}

            Don't recommend any more stocks. If the user wants new
            recommendations, instruct them to refresh the application.
            
            Continue to assist the user as you are able. If you don't have
            access to some information, let the user know. Keep answers concise
            so you can get rapid feedback from the user. Try to keep things
            amicable. Don't try to get fancy with formatting. The renderer is
            very simple. Try to avoid nesting formatting such as lists and
            line breaks inside tables. Of great importance is that the chat
            window is fairly small (about 400px width by 500px height). 
            
            Make sure your responses are conversational in nature, unless
            explicitly asked to do otherwise by the user.
            
            Finally, please avoid saying something which might offend the user.
            The descriptions of the profile clusters aren't flattering. 
        """,
    ))

    st.session_state.get("chat_history").append(AIMessage(
        content="I'm here to help! Do you have any questions about my "
                "recommendations?",
    ))
    return None

def render_stock(stock: StockInfo):
    st.subheader(stock.name)
    info_tuples = [
        ("Ticker", stock.ticker_symbol, "left"),
        ("Price", stock.price, "center"),
        ("Price to Earnings", stock.price_to_earnings, "right"),
    ]
    for (label, data, alignment), col in zip(info_tuples, st.columns(3)):
        if data is not None:
            col.markdown(f"""
                <div style='text-align: {alignment};'>
                    {label}: {data}
                </div>
                """,
                 unsafe_allow_html=True,
             )

    if stock.history is not None:
        st.plotly_chart(px.line(stock.history, y="Price"))

    with st.expander("Business Summary"):
        st.write(stock.description)

def render_chat_open_button():
    hidden = """
        display: none !important;
    """
    displayed = """
        position: fixed;
        bottom: 15px;
        right: 15px;
        z-index: 9999;
        width: fit-content;
    """

    container = st.container()
    with container:
        if st.button("AI Assistant"):
            st.session_state["is_chat_open"] = True
            st.rerun()
    css = displayed if not st.session_state.get("is_chat_open", True) else hidden
    container.float(css)

def render_chat():
    hidden = """
        display: none !important;
    """
    displayed = """
        position: fixed;
        bottom: 15px;
        right: 15px;
        width: 450px;
        max-height: 80vh;
        background-color: hsl(from var(--default-backgroundColor) h s l / 0.95);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        padding: 10px;
        overflow: hidden;
    """

    # Inject CSS
    st.markdown("""
            <style>
                div[data-testid="stChatMessage"] {
                    padding: 0;
                }
                div[data-testid="stVerticalBlock"]:has(div[data-testid="stChatMessage"]) div[data-testid="stChatMessage"]{
                    gap: 0.5rem;
                }
            </style>
        """, unsafe_allow_html=True)

    prepare_chat()

    chat_hist = st.session_state.get("chat_history")

    container = st.container()
    with container:
        title_bar, button_bar = st.columns([0.9, 0.10])
        title_bar.markdown("#### AI Assistant")
        if button_bar.button("—"):
            st.session_state["is_chat_open"] = False
            st.rerun()

        messages_container = st.container(height=450, border=False)

        with messages_container:
            for message in chat_hist:
                match message:
                    case SystemMessage():
                        pass
                    case AIMessage():
                        st.chat_message("assistant").write(message.content)
                    case HumanMessage():
                        st.chat_message("user").write(message.content)

        # Font-Awesome icons are annoying to render with Streamlit.
        if user_message := st.chat_input("⌨️"):
            with messages_container:
                st.chat_message("user").write(user_message)
                chat_hist.append(HumanMessage(content=user_message))

                llm = ChatOpenAI(
                    **get_llm_provider_env_dict(getenv("DEFAULT_LLM_PROVIDER")),
                    temperature=0.6,
                )

                with st.chat_message("assistant"):
                    response = st.write_stream(llm.stream(chat_hist))

                    chat_hist.append(AIMessage(content=response))

    css = displayed if st.session_state.get("is_chat_open", True) else hidden
    container.float(css)

def render_start_over_button():
    css = """
        position: fixed;
        bottom: 15px;
        left: 15px;
        z-index: 9999;
        width: fit-content;
    """

    container = st.container()
    with container:
        if st.button("Start Over", on_click=reset_session_state):
            st.rerun()

    container.float(css)


def render_recommendation_stage():

    st.title("Your Recommendations")

    for stock in st.session_state.get("stocks"):
        st.divider()
        render_stock(stock)

    render_start_over_button()

    if "is_chat_open" not in st.session_state:
        st.session_state["is_chat_open"] = True
    render_chat_open_button()
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