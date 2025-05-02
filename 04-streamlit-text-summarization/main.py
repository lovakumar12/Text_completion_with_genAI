import streamlit as st
#from langchain_openai import OpenAI
from euriai import EuriaiLangChainLLM
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")


llm = EuriaiLangChainLLM(
api_key=api_key,
model="gemini-2.0-flash-001",
temperature=0.7,
max_tokens=300
)


def generate_response(txt,llm):
    # llm = OpenAI(
    #     temperature=0,
    #     openai_api_key=openai_api_key
    # )
    llm=llm
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.invoke(docs)


st.set_page_config(
    page_title = "Writing Text Summarization"
)
st.title("Writing Text Summarization")



txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)



# result = []
# with st.form("summarize_form", clear_on_submit=True):
#     openai_api_key = st.text_input(
#         "OpenAI API Key",
#         type="password",
#         disabled=not txt_input
#     )
#     submitted = st.form_submit_button("Submit")
#     if submitted and openai_api_key.startswith("sk-"):
#         response = generate_response(txt_input)
#         result.append(response)
#         del openai_api_key
result = []
response = generate_response(txt_input,llm)
result.append(response)


if len(result):
    st.info(response)