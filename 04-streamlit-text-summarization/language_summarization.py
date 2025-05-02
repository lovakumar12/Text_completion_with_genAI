import streamlit as st
from euriai import EuriaiLangChainLLM
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize the LLM
llm = EuriaiLangChainLLM(
    api_key=api_key,
    model="gemini-2.0-flash-001",
    temperature=0.7,
    max_tokens=500
)

# Language options
language_options = ["English", "Hindi", "Telugu", "Tamil", "Spanish", "French", "German", "Other"]

# Streamlit UI
st.set_page_config(page_title="Multilingual Text Summarizer")
st.title("🌐 Multilingual Text Summarizer")

# Select input language
input_lang = st.selectbox("Select the **input** language:", language_options)

# Select output language
output_lang = st.selectbox("Select the **summary** output language:", language_options)

# Input text area
user_input = st.text_area(
    f"Enter your text in {input_lang}:",
    height=300
)

# Function to summarize and translate
def summarize_and_translate(text, input_lang, output_lang, llm):
    # Step 1: Split the text into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Step 2: Create summarization chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True
    )

    # Step 3: Generate summary in input language
    summary = chain.run(docs)

    # Step 4: Translate summary if needed
    if input_lang != output_lang:
        translation_prompt = f"Translate the following summary from {input_lang} to {output_lang}:\n\n{summary}"
        summary = llm.invoke(translation_prompt)

    return summary

# Button to generate summary
if st.button("Generate Summary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            result = summarize_and_translate(user_input, input_lang, output_lang, llm)
        st.subheader(f"📝 Summary in {output_lang}:")
        st.info(result)
