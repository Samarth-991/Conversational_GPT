import os
import random
import string
from numpy.random import randint
import streamlit as st
from streamlit_chat import message
from signal_handler.constant import EMBEDDING, LOCAL_VECTOR_DB ,CHAT_HISTORY
from backend.core import run_llm
import openai

st.title("DAMAC Conversation AI Bot")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

def summarize_chat_history(raw_txt):
    text_sum = "Summarize chat history in points :\n\n" + raw_txt
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=text_sum,
        temperature=0.1,  ## Generate randomness in the output
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = completion.choices[0].text
    return response

def generate_random_key(length=3):
    result = ''.join((random.choice(string.ascii_lowercase) for x in range(length)))
    return result

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
            if CHAT_HISTORY:
                summarized_chat_history = summarize_chat_history(raw_txt=generated_response['answer'])
                st.session_state["chat_history"].append((prompt, summarized_chat_history)) #generated_response['answer']
            else:
                st.session_state["chat_history"] = []


        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(st.session_state["chat_answers_history"],
                                                      st.session_state["user_prompt_history"], ):
                message(user_query, is_user=True,key=generate_random_key(length=randint(1,5)))
                message(generated_response,key=generate_random_key(length=randint(1,5)))

col1, col2 = st.columns(2)

with col2:
    reset_button = st.button("Reset")
    if reset_button:
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answers_history"] = []
        st.session_state["chat_history"] = []