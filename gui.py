import sys
import warnings
import os


def silence_streamlit_torch_errors():
    import logging
    logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    sys.stderr = open('/dev/null', 'w')

silence_streamlit_torch_errors()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import streamlit as st
st.set_page_config(page_title="RAG Question Answering", layout="wide")
from app import get_qa_chain

# Initialize session state
if 'qa_chain' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state.qa_chain = get_qa_chain()

# Title
st.title("RAG Question Answering System")

# Question input
query = st.text_input("Enter your question:", placeholder="What is a spam message?")

# Process question when submitted
if query:
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": query})
            
            # Display answer
            st.subheader("Answer")
            st.write(result['result'])
            
            # Display sources
            st.subheader("Sources")
            for doc in result['source_documents']:
                st.markdown(f"- {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}") 