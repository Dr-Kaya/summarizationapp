#text summarization
# Install the packages
import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

#generate response
def generate_response(txt):
    #instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    #split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    #create multiple documents
    docs = [Document(page_content=t) for t in texts]
    #Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

#page title
st.set_page_config(page_title="Text Summarization App")
st.title("Text Summarization App")

#Text input
txt_input = st.text_area("Enter the text to be summarized", height=200)

#Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit = True):
    openai_api_key = st.text_input("Enter your OpenAI API key", value="", type="password", disabled = not txt_input)
    submitted = st.form_submit_button('Summarize')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
    
