import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
import dotenv
dotenv.load_dotenv()

st.title("Chatbot")
st.write("Welcome to the chatbot! How can I assist you today?")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "./bella_vista_restaurant.pdf"
    loader = PyPDFLoader(pdf_name)
    index = VectorstoreIndexCreator(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        ),
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
    ).from_loaders([loader])
    return index.vectorstore

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a restaurant assistant. Answer using ONLY the context below.
- If the item is mentioned anywhere, confirm it and share details.
- If partial info exists, share what you know.
- If the exact item is NOT found, respond like this:
    "We don't have [item] on our menu, but you might enjoy these similar options:"
    Then list 2-3 relevant dishes from the context that are similar in category or ingredients.
- Never say "I don't know." Always try to suggest alternatives from the context, if user asks question related to the context.
- Only say "I don't know i can assist you only [context]" if the topic is completely absent from context like "What is the capital of France?".
- Be friendly and conversational.


Context:
{context}

Question: {question}

Answer:"""
)

prompt = st.chat_input("Type your message here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_chat = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    try:
        vectorstore = get_vectorstore()

        if vectorstore is None:  
            st.error("Failed to load doc")
            st.stop()

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10,  # retrieve more chunks
                                                                 "score_threshold": 0.3})  # accept even loosely related chunks
        retriever = MultiQueryRetriever.from_llm(  # rephrases query for better hits
            retriever=base_retriever,
            llm=groq_chat
        )

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=retriever,  # multi-query retriever
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt} 
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")