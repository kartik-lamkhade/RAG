import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

st.title("Document Q&A")

# Cache models with API token
@st.cache_resource
def load_models():
    parser = StrOutputParser()
    
    # Get token from Streamlit secrets
    hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN", None)
    
    model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            huggingfacehub_api_token=hf_token  # Add token here
        )
    )
    emb_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return parser, model, emb_model

parser, model, emb_model = load_models()

# File uploader
file = st.file_uploader("Upload file here", type=['pdf'])

# Process button
if st.button("start conversation"):
    if file is not None:
        try:
            with st.spinner("Processing PDF..."):
                # Get file bytes
                file_bytes = file.getvalue()
                st.info(f"File uploaded: {file.name}, Size: {len(file_bytes)} bytes")
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_path = tmp.name
                
                # Load PDF
                loader = PDFPlumberLoader(temp_path)
                document = loader.load()
                os.unlink(temp_path)
                
                st.success(f"✅ Loaded {len(document)} page(s)")
                
                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = []
                for d in document:
                    docs.extend(splitter.split_text(d.page_content))
                
                documents = [Document(page_content=doc) for doc in docs]
                
                # Create vector store
                vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=emb_model
                )
                
                st.session_state.vector_store = vector_store
                st.session_state.processed = True
                st.success(f"✅ Vector store created with {len(documents)} chunks!")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show details"):
                st.code(traceback.format_exc())
    else:
        st.error("❌ Please upload a PDF file first!")

# Query interface
if 'processed' in st.session_state and st.session_state.processed:
    st.markdown("---")
    st.subheader("💬 Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and query:
        try:
            with st.spinner("Generating answer..."):
                # Similarity search
                output = st.session_state.vector_store.similarity_search(query, k=5)
                context = "\n\n".join([doc.page_content for doc in output])
                
                # Create prompt
                template = PromptTemplate(
                    template="""Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:""",
                    input_variables=["context", "question"]
                )
                chain = template | model | parser
                
                # Get answer
                answer = chain.invoke({"context": context, "question": query})
                
                st.success("✅ Answer:")
                st.write(answer)
                
                with st.expander("📄 View Retrieved Context"):
                    st.text(context)
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show details"):
                st.code(traceback.format_exc())
