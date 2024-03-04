# import os
import os
import tempfile

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_extras.add_vertical_space import add_vertical_space


@st.cache_resource(ttl="1h")
def load_retriever(pdf_files):
    """load pdf files"""

    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for pdf_file in pdf_files:
        temp_pdf_file_path = os.path.join(temp_dir.name, pdf_file.name)

        with open(temp_pdf_file_path, "wb") as f:
            f.write(pdf_file.getvalue())

        loader = PyPDFLoader(temp_pdf_file_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # embeddings
    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5},
    )

    return retriever


def main():
    """main"""

    st.set_page_config(
        page_title="Talk to PDF using GPT 3.5",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.header("Talk to PDF files 📰", divider="rainbow")
    st.subheader(
        "Enjoy :red[talking] with :green[PDF] files using :sunglasses: OpenAI GPT 3.5 Turbo"
    )

    st.sidebar.title("Talk to PDF 📰")
    st.sidebar.markdown(
        "[Checkout the repository](https://github.com/ThivaV/chat_with_pdf_using_gpt)"
    )
    st.sidebar.markdown(
        """
            ### This is a LLM powered chatbot, built using:
                
            * [Streamlit](https://streamlit.io)
            * [LangChain](https://python.langchain.com/)
            * [OpenAI](https://platform.openai.com/docs/models)
            ___
            """
    )

    add_vertical_space(2)

    openai_key = st.sidebar.text_input(label="Enter the OpenAI key 👇", type="password")

    if not openai_key:
        st.info("👈 :red[Please enter the OpenAI key] ⛔")
        st.stop()

    # set the OPENAI_API_KEY to environment
    os.environ["OPENAI_API_KEY"] = openai_key

    add_vertical_space(1)

    upload_pdf_files = st.sidebar.file_uploader(
        "Upload a pdf files 📤", type="pdf", accept_multiple_files=True
    )

    if not upload_pdf_files:
        st.info("👈 :red[Please upload pdf files] ⛔")
        st.stop()

    retriever = load_retriever(upload_pdf_files)

    chat_history = StreamlitChatMessageHistory()

    # init chat history memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=chat_history, return_messages=True
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_key,
        temperature=0,
        streaming=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=False
    )

    # load previous chat history
    # re-draw the chat history in the chat window
    for message in chat_history.messages:
        st.chat_message(message.type).write(message.content)

    if prompt := st.chat_input("Ask questions"):
        with st.chat_message("human"):
            st.markdown(prompt)

        response = chain.run(prompt)

        with st.chat_message("ai"):
            st.write(response)


if __name__ == "__main__":
    # init streamlit
    main()
