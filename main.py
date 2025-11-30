import streamlit as st
import time

# STAGE - SURVEY

if "stage" not in st.session_state:
    st.session_state["stage"] = "survey"

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {}

def reset_session_state():
    st.session_state["stage"] = "survey"
    st.session_state["user_profile"] = {}



def make_survey_question(question_name, answers):
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

            # HARD-CODED UNTIL SURVEY CLUSTER MAPPING IS SET UP.
            st.session_state["cluster_result"] =\
                "Frugal, low-profile, and cautious."

            st.session_state["stage"] = "recommending"
            st.rerun()


def render_recommending_phase():
    st.title("Generating recommendations...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        "Mapping user profile to clusters...",
        "Identifying suitable tickers...",
        "Fetching live data from Yahoo Finance...",
        "Drafting recommendations..."
    ]

    # MOCK GENERATION STAGES WHILE NOT FULLY IMPLEMENTED
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.8)

    st.session_state["stage"] = "chat"
    st.rerun()



def render_agent_chat():

    st.title("Your Recommendations")
    st.info("Tickers and charts will appear here.")

    st.subheader("Chat")
    st.chat_message("assistant").write(
        "This is placeholder text. In the future, I'll be able to have a full "
        "conversation with you!")

    if prompt := st.chat_input("Converse with the AI agent..."):
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write("Sorry, but I'm not yet an AI agent!")

    st.button("Start Over", on_click=reset_session_state)



if st.session_state['stage'] == 'survey':
    render_survey()

elif st.session_state['stage'] == 'recommending':
    render_recommending_phase()

elif st.session_state['stage'] == 'chat':
    render_agent_chat()