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

# Page config 
st.set_page_config(
    page_title="Bella Vista Assistant",
    page_icon="🍽️",
    layout="centered"
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

/*  Root variables  */
:root {
    --bg:        #0f0e0c;
    --surface:   #1a1814;
    --border:    #2e2b26;
    --gold:      #c9a84c;
    --gold-dim:  #8a6f32;
    --cream:     #f0e6d0;
    --muted:     #7a7060;
    --user-bg:   #1f1d18;
    --bot-bg:    #16150f;
    --radius:    14px;
}

/*  Global reset  */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stBottom"] { background: var(--bg) !important; }

/*  Hide default streamlit chrome  */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/*  Header  */
.bv-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.bv-logo {
    font-size: 2.6rem;
    margin-bottom: 0.2rem;
}
.bv-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.03em;
    margin: 0;
}
.bv-subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/*  Status pill  */
.bv-status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1a2418;
    border: 1px solid #2a3d26;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.72rem;
    color: #6dbd5a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.bv-status-dot {
    width: 6px; height: 6px;
    background: #6dbd5a;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/*  Chat messages  */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background: var(--user-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) 4px var(--radius) !important;
    padding: 0.75rem 1rem !important;
    color: var(--cream) !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    background: var(--bot-bg) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--gold) !important;
    border-radius: 4px var(--radius) var(--radius) var(--radius) !important;
    padding: 0.75rem 1rem !important;
    color: var(--cream) !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

/* Avatar icons */
[data-testid="stChatMessageAvatarUser"] {
    background: var(--gold-dim) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 50% !important;
    filter: brightness(1.1) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--gold) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 50% !important;
    filter: brightness(1.1) !important;
}

/*  Chat input  */
[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: 0 0 0 0 transparent;
    transition: border 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: var(--gold) !important;
}

/*  Spinner  */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/*  Error box  */
[data-testid="stAlert"] {
    background: #1f1010 !important;
    border: 1px solid #5c2020 !important;
    border-radius: var(--radius) !important;
    color: #e07070 !important;
}


</style>
""", unsafe_allow_html=True)

# Header 
st.markdown("""
<div class="bv-header">
    <div class="bv-logo">🍽️</div>
    <p class="bv-title">Bella Vista</p>
    <p class="bv-subtitle">Restaurant Assistant</p>
    <div style="display:flex; justify-content:center; margin-top:0.5rem;">
        <span class="bv-status">
            <span class="bv-status-dot"></span>
            Online &nbsp;·&nbsp; Ready to help
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Session state for chat history 
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history 
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Load and vectorize PDF document (cached)
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

# Custom prompt for RetrievalQA chain (system instructions for the assistant)
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

# User Chat input 
prompt = st.chat_input("Ask anything about Bella Vista...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_chat = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    try:
        with st.spinner(""):
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load document. Please check the PDF path.")
                st.stop()

            # retrieve more chunks & accept even loosely related chunks
            base_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10, "score_threshold": 0.3}
            )
            # Rephrases query in multiple ways for better recall
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=groq_chat
            )
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt}
            )
            result = chain({"query": prompt})
            response = result["result"]

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Something went wrong: {e}")