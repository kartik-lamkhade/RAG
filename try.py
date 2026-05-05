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

# Cache models
@st.cache_resource
def load_models():
    parser = StrOutputParser()
    model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-7B-Instruct"
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
                # METHOD 1: Write file properly
                # Get the file bytes first
                file_bytes = file.getvalue()
                st.info(f"File uploaded: {file.name}, Size: {len(file_bytes)} bytes")
                
                # Create temp file and write content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
                    tmp.write(file_bytes)  # Write the bytes!
                    tmp.flush()  # Force write to disk
                    temp_path = tmp.name
                
                # Verify file was written
                if os.path.exists(temp_path):
                    file_size = os.path.getsize(temp_path)
                    st.info(f"Temp file created: {temp_path}, Size: {file_size} bytes")
                    
                    if file_size == 0:
                        st.error("❌ Error: Temp file is empty!")
                        os.unlink(temp_path)
                        st.stop()
                else:
                    st.error("❌ Error: Temp file was not created!")
                    st.stop()
                
                # Load PDF
                st.info("Loading PDF...")
                loader = PDFPlumberLoader(temp_path)
                document = loader.load()
                
                # Clean up temp file
                os.unlink(temp_path)
                
                st.success(f"✅ Loaded {len(document)} page(s)")
                
                # Split documents
                st.info("Splitting document into chunks...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = []
                for d in document:
                    docs.extend(splitter.split_text(d.page_content))
                
                documents = [Document(page_content=doc) for doc in docs]
                st.info(f"Created {len(documents)} chunks")
                
                # Create vector store
                st.info("Creating vector store...")
                vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=emb_model
                )
                
                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.processed = True
                st.success(f"✅ Vector store created with {len(documents)} chunks!")
                
        except Exception as e:
            st.error(f"❌ Error Type: {type(e).__name__}")
            st.error(f"❌ Error Message: {str(e)}")
            
            import traceback
            with st.expander("🔍 Show full error details"):
                st.code(traceback.format_exc())
    else:
        st.error("❌ Please upload a PDF file first!")

# Query interface (shown only after processing)
if 'processed' in st.session_state and st.session_state.processed:
    st.markdown("---")
    st.subheader("💬 Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and query:
        try:
            with st.spinner("Searching and generating answer..."):
                # Similarity search
                output = st.session_state.vector_store.similarity_search(query, k=5)
                
                # Combine context
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
                    for i, doc in enumerate(output, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show details"):
                st.code(traceback.format_exc())
