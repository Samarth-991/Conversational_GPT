import streamlit as st
from streamlit_chat import message

from backend.core import run_llm
from signal_handler.process_signal import ConfigParser as Parser

st.title("DAMAC Conversation AI Bot")

cnf_path = '/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf'
parser = Parser(cnf_path)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

with st.form(key='prompt', clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="Enter Your prompt here ...", key="1")
    submit_button = st.form_submit_button("Submit")
    reset_button = st.form_submit_button("Reset")
    if submit_button:
        with st.spinner("Generating Response ..."):
            generated_response = run_llm(query=prompt,
                                         vector_store=parser.get_vectorstore_attributes()['local_vector_store'],
                                         chat_history=st.session_state["chat_history"],
                                         embedding_model=parser.get_vectorstore_attributes()['embedding']
                                         )

            formatted_response = f"{generated_response['answer']} \n\n"
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append((prompt, generated_response['answer']))

        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                                      st.session_state["user_prompt_history"], ):
                message(user_query, is_user=True)
                message(generated_response)
    if reset_button:
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answers_history"] = []
        st.session_state["chat_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answers_history"] = []
