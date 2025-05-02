import streamlit as st
#from langchain_openai import OpenAI
from euriai import EuriaiLangChainLLM
from langchain_core.prompts import PromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")


st.set_page_config(
    page_title = "Blog Post Generator"
)

st.title("Blog Post Generator")

# openai_api_key = st.sidebar.text_input(
#     "OpenAI API Key",
#     type = "password"
# )

def generate_response(topic):
    #llm = OpenAI(openai_api_key=openai_api_key)
    llm = EuriaiLangChainLLM(
    api_key=api_key,
    model="gpt-4.1-nano",
    temperature=0.7,
    max_tokens=300
)

    
    template = """
    As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """
    prompt = PromptTemplate(
        input_variables = ["topic"],
        template = template
    )
    query = prompt.format(topic=topic)
    #response = llm(query, max_tokens=2048)
    response = llm(query)
    return st.write(response)

topic_text = st.text_input("Enter topic: ")
generate_response(topic_text)




# topic_text = st.text_input("Enter topic: ")
# if not openai_api_key.startswith("sk-"):
#     st.warning("Enter OpenAI API Key")
# if openai_api_key.startswith("sk-"):
#     generate_response(topic_text)
        
