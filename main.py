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


################
# Initial load
################
st_f.float_init()

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
            st.session_state["user_profile"]["responses"] = {
                "q0": q0,
                "q1": q1,
                "q2": q2,
                "q3": q3,
            }

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
        **get_llm_provider_env_dict(getenv("DEFAULT_PROVIDER")),
        temperature=0.6,
    )

    st.title("Generating recommendations...")
    progress_bar = st.progress(0)


    # Step 1 - Determine cluster
    progress_bar.progress(25, "Mapping responses...")

    with open("clusters.json", "r") as clusters_file:
        clusters = load_json(clusters_file)

    # REPLACE WITH CLUSTER MAPPING WHEN READY.
    st.session_state["user_profile"]["cluster_result"] = sample(clusters, 1)[0]

    # Step 2 - Get AI recommendation
    progress_bar.progress(50, "Consulting recommender...")

    cr = st.session_state.get("user_profile").get("cluster_result")

    # We should consider loading this from another file.
    st.session_state.get("chat_history").append(SystemMessage(content=f"""
        You are an AI agent meant to recommend tickers to check out for users
        based on clustering profile from a short survey.
        
        This user's result was:
        - Profile: {cr["profile"]}
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
                    **get_llm_provider_env_dict(getenv("DEFAULT_PROVIDER")),
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