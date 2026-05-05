import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
token1 = st.secrets["GROQ_API_KEY"]

st.title("Document answer")
parser = StrOutputParser()
model = ChatGroq(
    groq_api_key=token1,
    model_name="llama3-8b-8192"
)
emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
file = st.file_uploader("Uplode file here",type='pdf')
if "vectore_store" not in st.session_state:
    st.session_state.vectore_store = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if st.button("start coversation"):
    if file is not None:
        try:
            file_bytes = file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                temp_path = tmp.name
            loader = PDFPlumberLoader(temp_path)
            document = loader.load()
            spliter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
            docs = []
            for d in document:
                docs.extend(spliter.split_text(d.page_content))
            documents = [Document(page_content=doc) for doc in docs]
            vectore_store = FAISS.from_documents(
                documents=documents,
                embedding=emb_model
            )
            st.session_state.vectore_store = vectore_store
            st.session_state.processed = True
        except Exception as e:
            st.write(f"Error {e}")

if st.session_state.processed:     
    Query = st.text_input("Enter query here")
    if st.button("predict"):
        if Query:
            try:
                with st.spinner("Generating answer"):
                    output = st.session_state.vectore_store.similarity_search(Query,k=5)
                    def add(doc):
                        a = ""
                        for i in doc:
                            a += i.page_content + "\n"
                        return a
                    a = add(output)
                    template = PromptTemplate(template="give me answer for the question based on the following context: {context} and question: {question}",
                                            input_variables=["context","question"])
                    chain = template | model | parser
                    out = chain.invoke({"context":a,"question":Query})
                    st.write(out)
            except Exception as e:
                st.write(f"Error {e}")
        else:
            st.write("noooooooooooooooooo")
else:
    st.write("NUN")








