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
from langchain.schema import HumanMessage, AIMessage
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

    def create_qdrant_collection(qdrant_client, collection_name, vector_size):
        existing_collections = [col.name for col in qdrant_client.get_collections().collections]
        if collection_name in existing_collections:
            st.warning(f"Collection '{collection_name}' already exists.")
            return

        st.success("Creating Qdrant Collection for Embeddings")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks):
        create_qdrant_collection(qdrant_client, collection_name, len(embeddings[0]))
        
        st.success("Storing Embeddings in Qdrant")
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
    collection_name = "new_documents_practice"
    umique_id = st.text_input("Enter a Uique Hospital ID/Name:")
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
    # Initialize session state for chat history and unique_id tracking
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "last_unique_id" not in st.session_state:
        st.session_state.last_unique_id = None
        
    def query_embedding(query, api_key):
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return embeddings_model.embed_query(query)

    def search_related_text(query_embedding,unique_id, collection_name, top_k=3):
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            search_params=SearchParams(hnsw_ef=128),
            limit=top_k
        )
        return [
            result.payload["text"]
            for result in search_results.points
            if result.payload.get("unique_id") == unique_id
        ]

    def generate_response(llm, related_texts, user_query):
        memory = st.session_state.chat_history  # Use session memory
        conversation_history = st.session_state.chat_history.chat_memory.messages
    
        # conversation_history = memory.chat_memory.messages
        formatted_history = "\n".join([
            f"User: {message.content}" if isinstance(message, HumanMessage) else f"Assistant: {message.content}"
            for message in conversation_history
        ])
        
        st.success(f"{formatted_history}\n")
        if related_texts:
            formatted_text = "\n".join(related_texts)
            prompt = f"""
            You are an interactive assistant who answers questions in a friendly and conversational tone, just like a real person from the company. 
            Your responses should sound natural, concise, warm, and engaging—like you're having a chat with the user. 
            Imagine you're speaking directly to the user, just like how a colleague from the company would interact with them. 
            If there’s no answer, politely let the user know.
            
            Here’s the relevant information that you should keep in mind:
            {formatted_text}
    
            Here's the conversation history so far:
            {formatted_history}
    
            Now, answer the user's question in a way that makes them feel like they're talking to a real person from the company. Feel free to offer additional insights or ask follow-up questions if needed. The user's query is:
            {user_query}
            """
        else:
            prompt = f"""
            You are an interactive assistant who answers questions in a friendly and conversational tone, just like a real person from the company. 
            If there’s no relevant information to answer the user’s question, kindly inform them that you don’t have the necessary details and encourage them to ask something else.
    
            Unfortunately, we don't have relevant information available for your query at the moment. Please feel free to ask something else!
    
            Here's the conversation history so far:
            {formatted_history}
    
            Now, answer the user's question in a way that makes them feel like they're talking to a real person from the company. Feel free to offer additional insights or ask follow-up questions if needed. The user's query is:
            {user_query}
            """
    
        response = llm.invoke(prompt)
        return response.content.strip()

    # def collections_list(qdrant_client):
    def list_unique_ids_in_collection(qdrant_client, collection_name, limit=100):
        unique_ids = set()
        next_page_offset = None

        while True:
            points, next_page_offset = qdrant_client.scroll(
                collection_name=collection_name,
                with_payload=True,
                limit=limit,
                offset=next_page_offset,
            )

            for point in points:
                if "unique_id" in point.payload:
                    unique_ids.add(point.payload["unique_id"])
    
            if next_page_offset is None:
                break

        return list(unique_ids)

    def pipeline(api_key, qdrant_client, collection_name, user_query, unique_id, top_k=2):
        query_embeddings = query_embedding(user_query, api_key)
        related_texts = search_related_text(query_embeddings, unique_id, collection_name, top_k=top_k)
        # st.write(related_texts)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.6,
            google_api_key=api_key
        )
        response = generate_response(llm, related_texts, user_query)

        st.session_state.chat_history.chat_memory.add_user_message(user_query)
        st.session_state.chat_history.chat_memory.add_ai_message(response)
    
        return response  
        
    st.title("AI Query Pipeline")
    st.write("Looking for specific information? Type your question and select the Hospital ID (Name) to get results instantly!")

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    collection_name = "new_documents_practice"
    api_key = os.getenv("GOOGLE_API_KEY")

    # Fetch unique IDs for the dropdown
    with st.spinner("Fetching Hospitals Names..."):
        try:
            hospitals = list_unique_ids_in_collection(qdrant_client,collection_name)
            if not hospitals:
                hospitals = ["Hospital is not stored yet."]  # Fallback option if none are found
        except Exception as e:
            st.error(f"Error fetching hospitals ID: {e}")
            hospitals = ["Error fetching hospitals ID"]

    
    user_query = st.text_input("Enter your Query:")    
    unique_id = st.selectbox("Select Hospiatl ID/Name:", options=hospitals)

    # here change
    if unique_id != st.session_state.last_unique_id:
        st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.last_unique_id = unique_id 
        
    if st.button("Run Query"):
        if api_key and qdrant_client and collection_name and user_query:
            try:
                with st.spinner("Processing your query..."):
                    response = pipeline(api_key, qdrant_client, collection_name, user_query,unique_id,top_k=2)
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
