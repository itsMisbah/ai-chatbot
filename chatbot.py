import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()



st.title("Chatbot")
st.write("Welcome to the chatbot! How can I assist you today?")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Type your message here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    grok_system_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that provides concise and accurate answers to user queries.
                          answer the following question: {user_prompt}, no small talk, just the answer.""")

    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model = model
    )

    chain = grok_system_prompt | groq_chat | StrOutputParser()

    response = chain.invoke({"user_prompt": prompt})
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})