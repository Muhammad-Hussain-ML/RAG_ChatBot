import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client.http.models import VectorParams, Distance, SearchParams
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Define the Home Page
def home_page():
    st.title("Welcome to the RAG Document Explorer!")
    st.write("""
        This dashboard is designed to enhance your workflows with two cutting-edge tools:

        ### Features:
        1. **Upload PDFs for Advanced Storage**:  
           - Easily upload PDF files and process them for storage.

        2. **Query AI for Answers**:  
           - Ask questions and get useful responses based on the stored data.

        ### Usage Guide:
        - Navigate to **PDF to Qdrant** to upload and process your documents.
        - Switch to **Query AI** to interact with your data by entering specific queries.
        - This streamlined interface ensures an intuitive and seamless experience.

        ### Get Started:
        Use the navigation bar on the left to access the features. Start by uploading a PDF or querying the database!
    """)

# Define the PDF to Qdrant Page
def pdf_to_qdrant_page():
    def extract_text_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        pdf_text = "".join(page.extract_text() for page in reader.pages)
        st.success("Extracting Text")
        return pdf_text

    def split_text_into_chunks(text, chunk_size=800, chunk_overlap=160):
        st.success("Splitting Text into Chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        return text_splitter.create_documents([text])

    def generate_embeddings(chunks, api_key, model_name="models/text-embedding-004"):
        st.success("Generating Embeddings")
        os.environ["GOOGLE_API_KEY"] = api_key
        embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name)
        return [embeddings_model.embed_query(chunk.page_content) for chunk in chunks]

    # def create_qdrant_collection(qdrant_client, collection_name, vector_size):
    #     existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    #     if collection_name in existing_collections:
    #         qdrant_client.delete_collection(collection_name)
    #     qdrant_client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    #     )

    # def store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks, unique_id):
    #     create_qdrant_collection(qdrant_client, collection_name, len(embeddings[0]))
    #     points = [
    #         PointStruct(
    #             id=i, vector=embeddings[i],
    #             payload={"unique_id": unique_id, "chunk_id": i, "text": text_chunks[i].page_content}
    #         )
    #         for i in range(len(embeddings))
    #     ]
    #     qdrant_client.upsert(collection_name=collection_name, points=points)

    def store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks):
        st.success("Creating Embeddings Qdrant")
        vector_size = len(embeddings[0])
        existing_collections = [col.name for col in qdrant_client.get_collections().collections]
        if collection_name in existing_collections:
            st.error(f"Hospital ID '{collection_name}' already exists. Please! Enter Unique Hospiatl ID.")
            return False
        
        st.success("Storing Embeddings in Qdrant")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        points = [
            PointStruct(
                id=i, vector=embeddings[i],
                payload={ "chunk_id": i, "text": text_chunks[i].page_content}
            )
            for i in range(len(embeddings))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)

        return True
        
    st.title("PDF to Qdrant Embedding Pipeline")
    st.write("""
        Upload PDF documents, process their content into meaningful chunks,
        and store the resulting embeddings in the Qdrant database.
    """)

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    collection_name = st.text_input("Enter a Uique Hospital ID/Name:")
    run_pipeline = st.button("Run Pipeline")

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    api_key = os.getenv("GOOGLE_API_KEY")

    if run_pipeline:
        if uploaded_file and collection_name:
            try:
                with st.spinner("Processing the PDF..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    text_chunks = split_text_into_chunks(pdf_text)
                    embeddings = generate_embeddings(text_chunks, api_key)
                    success= store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks)
                if success:
                    st.success(f"Data successfully stored in Qdrant under the collection: '{collection_name}'.")
                else:
                    st.error(f"Failed to store data in Qdrant. Please check the collection name '{collection_name}' or try again.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all the required inputs.")

# Define the Query AI Page
def query_ai_page():
    def query_embedding(query, api_key):
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return embeddings_model.embed_query(query)

    def search_related_text(query_embedding, collection_name, top_k=3):
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            search_params=SearchParams(hnsw_ef=128),
            limit=top_k
        )
        return [result.payload["text"] for result in search_results.points]

    def generate_response(retriever, api_key, user_query):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.6, google_api_key=api_key)
        conversation = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, verbose=False)
        return conversation.run(user_query)

    def pipeline(api_key, qdrant_client, collection_name, user_query, top_k=3):
        query_embeddings = query_embedding(user_query, api_key)
        related_texts = search_related_text(query_embeddings, collection_name, top_k)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings_model,
            content_payload_key="text"
        )
        retriever = vector_store.as_retriever()
        return generate_response(retriever, api_key, user_query)

    st.title("AI Query Pipeline")
    st.write("Enter a query to interact with stored document embeddings.")

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    collection_name = "new_documents_collection"

    user_query = st.text_input("Enter your Query:")
    unique_id = st.text_input("Enter Unique Product ID:")
    
    if st.button("Run Query"):
        if api_key and qdrant_client and collection_name and user_query:
            try:
                with st.spinner("Processing your query..."):
                    response = pipeline(api_key, qdrant_client, collection_name, user_query)
                st.success("Query executed successfully!")
                st.write("Generated Response:", response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all the required inputs.")

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to Page:", ["Home", "Upload PDFs", "Query AI"])

# Render the selected page
if page == "Home":
    home_page()
elif page == "Upload PDFs":
    pdf_to_qdrant_page()
elif page == "Query AI":
    query_ai_page()
