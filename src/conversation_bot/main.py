import os
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from signal_handler.process_signal import ConfigParser as Parser
from backend import data_loader as loader
from src.conversation_bot.data_orchestrator.data_processing import DataOrchestrator
import time


st.title("DAMAC Conversation AI Bot")

cnf_path = '/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf'
parser = Parser(cnf_path)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if 'person_name' not in st.session_state:
    st.session_state['person_name'] = ""


def get_employee_customer_details():
    data_orchestrator = DataOrchestrator(cnf_path)
    xlx_data = data_orchestrator.read_data(parser.get_records()['file_path'])
    customer_list,employee_list = data_orchestrator.get_customer_employee_list(xlx_data,
                                                 customer_index=parser.get_records()['customer_index'],
                                                 employee_index=parser.get_records()['client_index']
                                                 )
    customer_list.insert(0,'<select>')
    employee_list.insert(0,'<select>')
    return customer_list , employee_list

customer = st.checkbox('Customer')
employee = st.checkbox("Employee")
customer_names , employee_names = get_employee_customer_details()

def load_embeddings_data(person_name,relation='employee'):
    conv_texts = None
    if relation == 'employee':
        conv_texts = loader.load_employee_conversations(person_name)
    if relation =='customer':
        conv_texts = loader.load_customer_conversations(customer_name=person_name)
    if conv_texts:
        loader.load_embeddings(conv_texts,embedder='openai')
    return

option = '<select>'
if employee:
    option = st.selectbox("Select Employee ..", tuple(employee_names))
elif customer:
    option = st.selectbox("Select Customer ", tuple(customer_names))

if st.session_state['person_name'] == "":
    if option != '<select>':
        st.session_state['person_name'] = option
        with st.spinner("Loading Embeddings ..."):
            if employee:
                load_embeddings_data(person_name=option,relation='employee')
            if customer:
                load_embeddings_data(person_name=option,relation='customer')
            time.sleep(3)

else:
    if st.session_state['person_name'] != option:
        # reset state
        st.session_state["user_prompt_history"] = []
        st.session_state['chat_history'] = []
        st.session_state["chat_answers_history"] = []
        st.session_state['person'] = option
        if employee:
            conv_texts = loader.load_employee_conversations(option)
        else:
            conv_texts = loader.load_customer_conversations(option)
        if conv_texts:
            with st.spinner("Loading Embeddings ..."):
                loader.load_embeddings(conv_texts, embedder='openai')
                time.sleep(3)

with st.form(key='prompt',clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="Enter Your prompt here ...",key="1")
    submit_button = st.form_submit_button("Submit")

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