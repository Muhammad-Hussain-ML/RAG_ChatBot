import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client.http.models import VectorParams, Distance, SearchParams
from qdrant_client.models import PointStruct, Filter, FieldCondition
from qdrant_client.models import MatchValue
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import os
from pymongo import MongoClient
from datetime import datetime

# Define the Home Page
def home_page():
    st.title("Welcome to RAG Hospital Assistant!")

    st.write("""
    Streamline hospital information retrieval with AI-powered tools:
    
    ### Features:
    1. **PDF Upload & Storage**  
       - Upload hospital documents, extract content, and store embeddings in Qdrant.
    
    2. **AI-Powered Querying**  
       - Retrieve hospital-related insights using intelligent search.
    
    3. **Query Analysis & Management**  
       - Track and optimize hospital-related queries stored in MongoDB.
    
    ### How It Works:
    - **PDF to Qdrant** → Upload and process hospital documents.  
    - **Query AI** → Get accurate responses from indexed hospital data.  
    - **Query Analysis** → Monitor and refine user interactions.
    
    Use the sidebar to navigate!
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

    def store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks, unique_id):
        create_qdrant_collection(qdrant_client, collection_name, len(embeddings[0]))
        
        st.success("Storing Embeddings in Qdrant")
        points = [
            PointStruct(
                id=i, vector=embeddings[i],
                payload={ "unique_id": unique_id, "chunk_id": i, "text": text_chunks[i].page_content}
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
    collection_name = "new_practice"
    unique_id = st.text_input("Enter a Uique Hospital ID/Name:")
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
                    success= store_embeddings_in_qdrant(qdrant_client, collection_name, embeddings, text_chunks, unique_id)
                if success:
                    st.success(f"Data successfully stored in Qdrant under the collection: '{collection_name}'.")
                else:
                    st.error(f"Failed to store data in Qdrant. Please check the collection name '{collection_name}' or try again.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all the required inputs.")
            
# Define the Query History Page
import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient

def query_history_page():
    st.title("Query History")

    # Cached MongoDB connection
    @st.cache_resource()
    def get_mongo_client():
        MONGO_URI = os.getenv("MONGO_URI")
        return MongoClient(MONGO_URI)

    client = get_mongo_client()
    db = client["query_logs"]
    collection = db["user_queries"]

    # Fetch unique IDs dynamically without caching
    def get_unique_ids():
        return collection.distinct("unique_id")

    # Dropdown to select unique_id (fetches fresh IDs on every run)
    unique_id = st.selectbox(
        "**Select Hospital Name or ID:**",
        options=["Select a hospital or ID..."] + get_unique_ids()
    )

    if unique_id and unique_id != "Select a hospital or ID...":
        # Fetch all queries related to the selected unique_id, sorted by latest
        queries = list(collection.find({"unique_id": unique_id}).sort("timestamp", -1))

        if queries:
            df = pd.DataFrame(queries).drop(columns=["_id"], errors="ignore")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No queries found for the selected hospital ID.")


# Define the Query AI Page
def query_ai_page():
    # Initialize session state for chat history and unique_id tracking
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "last_unique_id" not in st.session_state:
        st.session_state.last_unique_id = None

    MONGO_URI = os.getenv("MONGO_URI"),
    client = MongoClient(MONGO_URI)
    db = client["query_logs"] 
    collection = db["user_queries"]

    def query_embedding(query, api_key):
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return embeddings_model.embed_query(query)

    def search_related_text(query_embedding, unique_id, collection_name, top_k=3):
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,  # Ensure this is a list
            query_filter=Filter(
                must=[FieldCondition(key="unique_id", match=MatchValue(value=unique_id))]
            ),
            limit=top_k  # Fixed syntax error
        )
    
        return [
            result.payload.get("text", "No related text available")
            for result in search_results.points
        ]


    def generate_response(llm, related_texts, user_query):
        memory = st.session_state.chat_history  # Use session memory
        conversation_history = st.session_state.chat_history.chat_memory.messages
        # conversation_history = memory.chat_memory.messages
        formatted_history = "\n".join([
            f"User: {message.content}" if isinstance(message, HumanMessage) else f"Assistant: {message.content}"
            for message in conversation_history
        ])
        # st.success(f"Conversation history loaded:\n{formatted_history}\n")
        if related_texts:
            formatted_text = "\n".join(related_texts)
            prompt = f"""
            You are a hospital's interactive assistant, designed to answer queries related to this hospital in a professional, friendly, and helpful manner. 
            Imagine you're speaking directly to the user, just like how a hospital representative or colleague would interact with them—warm, clear, and helpful.
            Your responses should be **strictly based on the relevant hospital information provided**. 
            You must **only** answer questions related to this hospital. If a query is unrelated, politely inform the user.
        
            **Here’s the most relevant hospital-related information to use for answering the query:**
            {formatted_text}
        
            **If needed, here’s the conversation history to maintain context:**
            {formatted_history}
        
            Now, answer the user's question **primarily using the relevant hospital information above**.  
            - If the answer is found in the relevant text, respond concisely using that information.  
            - If no direct answer is available, use the conversation history to maintain context but do **not** make up information.  
            - If the query is **unrelated to the hospital**, respond with:  
              "I'm here to assist with hospital-related queries. If you have any questions about appointments, facilities, doctors, or medical services, I'd be happy to help!"  
        
            The user's query is:
            {user_query}
            """
        else:
            prompt = f"""
            You are a hospital's interactive assistant, designed to answer queries related to this hospital in a professional, friendly, and helpful manner.  
            Imagine you're speaking directly to the user, just like how a hospital representative or colleague would interact with them—warm, clear, and helpful.
            Since no relevant information is currently available, please politely inform the user.
        
            **Unfortunately, I don’t have specific information available for your query at the moment.**  
            However, I can help with general hospital-related topics such as appointments, doctors, facilities, and medical services.  
            Please feel free to ask something else related to the hospital!
        
            **Here's the conversation history so far (for maintaining context):**
            {formatted_history}
        
            Now, answer the user's question **only if it is related to this hospital**.  
            - If it is relevant, provide a professional and clear response.  
            - If it is **not related**, respond with:  
              "I'm here to assist with hospital-related queries. If you have any questions about appointments, facilities, doctors, or medical services, I'd be happy to help!"  
        
            The user's query is:
            {user_query}
            """
        response = llm.invoke(prompt)
        return  response.content.strip()
    
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

    def pipeline(api_key, qdrant_client, collection_name, user_query, unique_id, top_k=4):
        query_embeddings = query_embedding(user_query, api_key)
        related_texts = search_related_text(query_embeddings, unique_id, collection_name, top_k=top_k)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.6,
            google_api_key=api_key
        )
        response = generate_response(llm, related_texts, user_query)
        
        st.session_state.chat_history.chat_memory.add_user_message(user_query)
        st.session_state.chat_history.chat_memory.add_ai_message(response)
        return  response
        
    # st.title("AI Query Pipeline")
    # st.write("Looking for specific information? Type your question and select the Hospital ID (Name) to get results instantly!")

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    collection_name = "IICI_docs_EMR"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        st.error("Google API Key is missing! Please check your environment variables.")

    # Fetch unique IDs for the dropdown
    with st.spinner("Fetching Hospitals Names..."):
        try:
            unique_ids = list_unique_ids_in_collection(qdrant_client,collection_name)
            if not unique_ids:
                unique_ids = ["Hospital is not stored yet."]  # Fallback option if none are found
        except Exception as e:
            st.error(f"Error fetching hospitals ID: {e}")
            unique_ids = ["Error fetching hospitals ID"]

    # Custom CSS to optimize page layout
    st.markdown("""
         <style>
               .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 1rem;
                }
         </style>
    """, unsafe_allow_html=True)

    # user_query = st.text_input("Enter your Query:")    
    # unique_id = st.selectbox("Select Hospiatl ID/Name:", options=hospitals)

   
    # Page Content
    col1, col2 = st.columns([3, 1])  # Two columns with ratio (3:1)
    
    # Title and Unique ID dropdown placed side by side
    with col1:
        st.title("AI Query Chat")
    
    with col2:
        unique_id = st.selectbox(
            "**Select Hospital Name or ID:**",
            index=None,
            placeholder="Select a hospital or ID...",
            options=unique_ids
        )

     # here change
    if unique_id != st.session_state.last_unique_id:
        st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.last_unique_id = unique_id 

    if "messages" not in st.session_state:
      st.session_state.messages = []

    # Display the conversation history (user messages and assistant replies)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  

    # User inputs
    user_query = st.chat_input("Enter your Query:")
    
    # Button to run pipeline
    if user_query:    
    # if st.button("Run Query"):
        if google_api_key and qdrant_client and collection_name and user_query and unique_id:

            query_data = {
                "query": user_query,
                "unique_id": unique_id,
                "timestamp": datetime.utcnow()  
            }
            collection.insert_one(query_data)
            
    
            # Append user's message to chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user's message in the chat interface
            with st.chat_message("user"):
                st.markdown(user_query)

            try:
                with st.spinner("Processing your query..."):
                    response = pipeline(google_api_key, qdrant_client, collection_name, user_query,unique_id,top_k=4)

                assistant_reply = response  # This is the assistant's response
                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)

                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                # st.write("AI :", response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all the required inputs.")

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to Page:", ["Home", "Upload PDFs", "Query AI", "Query Analysis"])

# Render the selected page
if page == "Home":
    home_page()
elif page == "Upload PDFs":
    pdf_to_qdrant_page()
elif page == "Query AI":
    query_ai_page()
elif page == "Query Analysis":
    query_history_page()

    
