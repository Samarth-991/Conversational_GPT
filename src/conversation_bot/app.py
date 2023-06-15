import os
import numpy as np
import streamlit as st
from streamlit_chat import message
from signal_handler.constant import EMBEDDING, LOCAL_VECTOR_DB, OPENAPI_KEY
from backend.core import run_llm

os.environ['OPENAI_API_KEY'] = OPENAPI_KEY

st.title("DAMAC Conversation AI Bot")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


with st.form(key='prompt', clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="Enter Your prompt here ...", key="1")
    submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        with st.spinner("Generating Response ..."):
            generated_response = run_llm(query=prompt,
                                         vector_store=LOCAL_VECTOR_DB,
                                         chat_history=st.session_state["chat_history"],
                                         embedding_model=EMBEDDING
                                         )
            formatted_response = f"{generated_response['answer']} \n\n"
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append((prompt, generated_response['answer']))

        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                                      st.session_state["user_prompt_history"], ):
                message(user_query, is_user=True,key=np.random.randint(1,50))
                message(generated_response,key=np.random.randint(50,100))

col1, col2 = st.columns(2)

with col2:
    reset_button = st.button("Reset")
    if reset_button:
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answers_history"] = []
        st.session_state["chat_history"] = []