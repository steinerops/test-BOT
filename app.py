import os
import streamlit as st
import tempfile
import logging
import requests
import base64
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import StdOutCallbackHandler

# Fix for LangSmithTracer import error
try:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    has_langsmith = True
except ImportError:
    has_langsmith = False
    class LangChainTracer:
        def __init__(self, *args, **kwargs):
            pass

# Updated Pinecone import for version compatibility
import pinecone
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

# Import speech recognition for voice input
try:
    import speech_recognition as sr
    has_speech_recognition = True
except ImportError:
    has_speech_recognition = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langchain_processes.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pdf_chatbot")

# Voice functionality using ElevenLabs
class VoiceHandler:
    def __init__(self):
        self.api_key = st.secrets.get("ELEVENLABS_API_KEY", "")
        self.voice_id = st.secrets.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs API"""
        if not self.api_key:
            logger.warning("ElevenLabs API key not found")
            return None
            
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling ElevenLabs API: {str(e)}")
            return None
    
    def get_available_voices(self):
        """Get list of available voices from ElevenLabs"""
        if not self.api_key:
            return []
            
        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                voices = response.json().get("voices", [])
                return [(voice["voice_id"], voice["name"]) for voice in voices]
            return []
        except Exception as e:
            logger.error(f"Error fetching voices: {str(e)}")
            return []

# Set up environment variables from Streamlit secrets
def setup_environment():
    # Set required API keys
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Set Pinecone environment for v2 compatibility
    if "PINECONE_ENVIRONMENT" in st.secrets:
        os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

    # Set LangSmith variables if they exist in secrets
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith tracing enabled")
    else:
        logger.info("LangSmith tracing not configured")

# Initialize environment
setup_environment()

# Initialize Pinecone with version detection
try:
    pinecone_version = pinecone.__version__.split('.')[0]
    logger.info(f"Detected Pinecone SDK version: {pinecone.__version__}")
    
    if int(pinecone_version) >= 4:
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        logger.info("Initialized Pinecone v4+ client")
    else:
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        )
        pc = pinecone
        logger.info("Initialized Pinecone v2-v3 client")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {str(e)}")
    st.error(f"Error initializing Pinecone: {str(e)}")
    pc = None

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class PDFChatbot:
    def __init__(self):
        logger.info("Initializing PDF Chatbot")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index_name = "pdf-ai"
        self.pc = pc

        callbacks = None
        if "LANGCHAIN_API_KEY" in os.environ and has_langsmith:
            try:
                tracer = LangChainTracer(
                    project_name=os.environ.get("LANGCHAIN_PROJECT", "pdf-chatbot")
                )
                callbacks = [StdOutCallbackHandler(), tracer]
                logger.info("LangSmith tracing enabled with callback handlers")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith tracing: {e}")
                callbacks = None

        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Successfully initialized LLM with gemini-1.5-flash model")
        except Exception as e:
            logger.warning(f"Error initializing gemini-1.5-flash: {e}")
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
                logger.info("Successfully initialized LLM with gemini-1.0-pro model")
            except Exception as e2:
                logger.warning(f"Error initializing gemini-1.0-pro: {e2}")
                self.llm = ChatGoogleGenerativeAI(model="models/gemini-pro")
                logger.info("Trying legacy model name format: models/gemini-pro")

        self._setup_pinecone_index()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self.callbacks = callbacks
        logger.info("PDF Chatbot initialization complete")

    def _setup_pinecone_index(self):
        if self.pc is None:
            logger.error("Pinecone client not initialized")
            st.error("Pinecone client not initialized")
            return
            
        try:
            pinecone_version = pinecone.__version__.split('.')[0]
            
            if int(pinecone_version) >= 4:
                index_names = [idx["name"] for idx in self.pc.list_indexes()]
                if self.index_name not in index_names:
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    try:
                        self.pc.create_index(
                            name=self.index_name,
                            dimension=384,
                            metric="cosine"
                        )
                        logger.info(f"Created new Pinecone index: {self.index_name}")
                    except Exception as e:
                        logger.error(f"Failed to create Pinecone index: {str(e)}")
                        st.error(f"Failed to create Pinecone index: {str(e)}")
                else:
                    logger.info(f"Using existing Pinecone index: {self.index_name}")
            else:
                index_list = self.pc.list_indexes()
                if self.index_name not in index_list:
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=384,
                        metric="cosine"
                    )
                    logger.info(f"Created new Pinecone index: {self.index_name}")
                else:
                    logger.info(f"Using existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error with Pinecone index setup: {e}")
            st.error(f"Error with Pinecone index setup: {e}")

    def extract_text_from_pdf(self, pdf_file):
        logger.info(f"Starting text extraction from PDF: {pdf_file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name

        logger.info(f"Temporary PDF file created: {pdf_path}")

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Extracted {len(documents)} pages from PDF")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            st.error(f"Error extracting text: {str(e)}")
            documents = []
        finally:
            os.unlink(pdf_path)
            logger.info(f"Temporary PDF file deleted")

        return documents

    def chunk_text(self, documents):
        logger.info("Starting text chunking process")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks from documents")

        if chunks:
            logger.info(f"Sample chunk content: {chunks[0].page_content[:100]}...")

        return chunks

    def create_embeddings_and_store(self, chunks, namespace):
        logger.info(f"Creating embeddings for {len(chunks)} chunks and storing in Pinecone namespace: {namespace}")

        try:
            pinecone_version = pinecone.__version__.split('.')[0]
            
            if int(pinecone_version) >= 4:
                index = self.pc.Index(self.index_name)
                try:
                    index.delete(namespace=namespace, delete_all=True)
                    logger.info(f"Deleted existing vectors in namespace: {namespace}")
                except Exception as e:
                    logger.warning(f"No existing vectors to delete or error: {str(e)}")
            else:
                index = self.pc.Index(self.index_name)
                try:
                    index.delete(deleteAll=True, namespace=namespace)
                    logger.info(f"Deleted existing vectors in namespace: {namespace}")
                except Exception as e:
                    logger.warning(f"No existing vectors to delete or error: {str(e)}")
                
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace
            )
            logger.info("Embeddings created and stored successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            
            try:
                logger.info("Attempting alternative approach for storing embeddings...")
                vectorstore = PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    index=index,
                    namespace=namespace
                )
                logger.info("Alternative approach successful")
                return vectorstore
            except Exception as e2:
                logger.error(f"Alternative approach also failed: {str(e2)}")
                raise e2

    def get_conversational_chain(self, vectorstore):
        logger.info("Creating conversational retrieval chain")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

        logger.info("Conversational retrieval chain created")
        return chain

    def process_query(self, chain, query):
        logger.info(f"Processing query: {query}")

        try:
            result = chain.invoke({"question": query})
            logger.info("Used chain.invoke() successfully")
        except AttributeError:
            result = chain({"question": query})
            logger.info("Used chain() call successfully")

        logger.info(f"Retrieved {len(result['source_documents'])} source documents")
        logger.info(f"Generated answer (first 100 chars): {result['answer'][:100]}...")

        return result["answer"], result["source_documents"]


def create_audio_player(audio_content):
    """Create an HTML audio player for the generated audio"""
    audio_base64 = base64.b64encode(audio_content).decode()
    audio_html = f"""
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def create_voice_input_interface():
    """Create voice input interface using Streamlit's built-in audio input"""
    st.markdown("### 🎤 Voice Input")
    
    # Use Streamlit's audio input widget
    audio_bytes = st.audio_input("Record your question")
    
    return audio_bytes

def process_audio_to_text(audio_input):
    """Convert audio to text using speech recognition"""
    if not has_speech_recognition:
        st.error("Speech recognition library not available. Please install speech_recognition.")
        return None
    
    if audio_input is None:
        return None
    
    try:
        # Initialize speech recognizer
        r = sr.Recognizer()
        
        # Handle both UploadedFile and bytes objects
        if hasattr(audio_input, 'getvalue'):
            # It's an UploadedFile object from Streamlit
            audio_bytes = audio_input.getvalue()
        else:
            # It's already bytes
            audio_bytes = audio_input
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            tmp_audio.write(audio_bytes)
            audio_file_path = tmp_audio.name
        
        try:
            # Load audio file and convert to text
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise and record audio
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.record(source)
                
            # Use Google's speech recognition
            text = r.recognize_google(audio)
            logger.info(f"Successfully converted audio to text: {text}")
            return text
            
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try speaking more clearly.")
            logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from speech recognition service: {e}")
            logger.error(f"Speech recognition service error: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_file_path)
                logger.info("Temporary audio file deleted")
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary audio file: {cleanup_error}")
                
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        logger.error(f"Error in process_audio_to_text: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="AI PDF Chatbot with Voice", layout="wide")
st.title("🎤 AI PDF Chatbot with Voice")
st.write("Upload a PDF and chat with it using voice or text!")

# Initialize session state
if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = PDFChatbot()
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        st.error(f"Error initializing chatbot: {str(e)}")
        st.session_state.chatbot = None

if "voice_handler" not in st.session_state:
    st.session_state.voice_handler = VoiceHandler()

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False

# Define the layout
left_col, right_col = st.columns([1, 2])

# Sidebar content
with left_col:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Voice settings
    st.header("🎤 Voice Settings")
    voice_enabled = st.toggle("Enable Voice Features", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = voice_enabled
    
    if voice_enabled and st.session_state.voice_handler.api_key:
        # Voice selection
        voices = st.session_state.voice_handler.get_available_voices()
        if voices:
            selected_voice = st.selectbox(
                "Select Voice",
                options=[voice[0] for voice in voices],
                format_func=lambda x: next((voice[1] for voice in voices if voice[0] == x), x),
                index=0
            )
            st.session_state.voice_handler.voice_id = selected_voice
        
        # Voice settings
        st.subheader("🔊 Voice Controls")
        auto_play = st.checkbox("Auto-play responses", value=True)
        
        # Speech recognition info
        if has_speech_recognition:
            st.success("✅ Speech recognition available")
        else:
            st.warning("⚠️ Speech recognition not available. Install speech_recognition library.")
        
    elif voice_enabled and not st.session_state.voice_handler.api_key:
        st.warning("⚠️ ElevenLabs API key not found. Voice features disabled.")
        st.info("Add your ElevenLabs API key to secrets to enable voice features.")

    debug_mode = st.toggle("Debug Mode", value=False)

    if "LANGCHAIN_API_KEY" in st.secrets:
        langsmith_url = "https://smith.langchain.com/projects/" + st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        st.markdown(f"[View Traces in LangSmith]({langsmith_url})")

    if uploaded_file and not st.session_state.document_processed and st.session_state.chatbot:
        with st.spinner("Processing document..."):
            status_container = st.empty()

            try:
                status_container.info("Extracting text from PDF...")
                documents = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)

                if not documents:
                    status_container.error("Failed to extract text from PDF. Please try another file.")
                else:
                    status_container.success(f"✅ Extracted {len(documents)} pages from PDF")

                    status_container.info("Chunking text...")
                    chunks = st.session_state.chatbot.chunk_text(documents)
                    status_container.success(f"✅ Created {len(chunks)} text chunks")

                    namespace = uploaded_file.name.replace(" ", "_").lower()

                    status_container.info("Creating embeddings and storing in Pinecone...")
                    vectorstore = st.session_state.chatbot.create_embeddings_and_store(chunks, namespace)
                    status_container.success("✅ Created embeddings and stored in Pinecone")

                    status_container.info("Setting up conversation chain...")
                    st.session_state.conversation = st.session_state.chatbot.get_conversational_chain(vectorstore)
                    st.session_state.document_processed = True
                    status_container.empty()
                    st.success("Document processed successfully!")
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                status_container.error(f"Error processing document: {str(e)}")

    if st.session_state.document_processed:
        st.subheader("📋 Document Information")
        st.info(f"📁 Filename: {uploaded_file.name}")

        if st.button("🔄 Process Another Document"):
            st.session_state.document_processed = False
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.rerun()

# Chat interface
with right_col:
    st.header("💬 Chat with your PDF")

    if not st.session_state.document_processed:
        st.info("Please upload a PDF document to start chatting.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                # Add audio player if voice is enabled and audio exists
                if voice_enabled and "audio" in message:
                    st.markdown(create_audio_player(message["audio"]), unsafe_allow_html=True)

        # Voice input section
        user_query = None
        
        if voice_enabled and has_speech_recognition:
            # Voice input using Streamlit's audio input
            audio_input = create_voice_input_interface()
            
            if audio_input is not None:
                with st.spinner("🎤 Converting speech to text..."):
                    voice_text = process_audio_to_text(audio_input)
                    if voice_text:
                        st.success(f"🎤 Voice input: {voice_text}")
                        user_query = voice_text

        # Text input (always available)
        if not user_query:
            user_query = st.chat_input("Ask a question about your document")

        # Process the query
        if user_query:
            st.chat_message("user").write(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("🤔 Thinking..."):
                try:
                    response, sources = st.session_state.chatbot.process_query(
                        st.session_state.conversation, user_query
                    )
                    
                    # Generate audio if voice is enabled
                    audio_content = None
                    if voice_enabled and st.session_state.voice_handler.api_key:
                        with st.spinner("🔊 Generating audio..."):
                            audio_content = st.session_state.voice_handler.text_to_speech(response)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    response = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
                    sources = []
                    audio_content = None

            # Display response
            st.chat_message("assistant").write(response)
            
            # Store response in chat history
            message_data = {"role": "assistant", "content": response}
            if audio_content:
                message_data["audio"] = audio_content
            st.session_state.chat_history.append(message_data)

            # Play audio if available
            if audio_content:
                st.markdown("🔊 **Audio Response:**")
                st.markdown(create_audio_player(audio_content), unsafe_allow_html=True)

            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i + 1}:**")
                        st.write(source.page_content)
                        st.write(f"📄 Page: {source.metadata.get('page', 'N/A')}")
                        st.divider()

# Setup instructions
if not st.session_state.document_processed:
    with st.expander("🔧 How to set up API keys and dependencies"):
        st.markdown("""
        ### Setting up your Streamlit Cloud secrets

        In your Streamlit Cloud dashboard:
        1. Go to your app settings
        2. Navigate to the "Secrets" section
        3. Add the following secrets:

        ```toml
        # Required API keys
        PINECONE_API_KEY = "your-pinecone-api-key"
        GOOGLE_API_KEY = "your-google-api-key"

        # Required for Pinecone client v2.x.x
        PINECONE_ENVIRONMENT = "gcp-starter"  # Or your specific environment

        # Voice functionality (ElevenLabs)
        ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
        ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Optional: specific voice ID

        # Optional LangSmith configuration (for advanced monitoring)
        LANGCHAIN_API_KEY = "your-langsmith-api-key"
        LANGCHAIN_PROJECT = "pdf-chatbot"
        LANGCHAIN_TRACING_V2 = "true"
        ```

        ### Requirements.txt for Streamlit Cloud
        
        Add this to your requirements.txt file:
        ```
        streamlit
        langchain
        langchain-community
        langchain-huggingface
        langchain-google-genai
        langchain-pinecone
        pinecone-client
        google-generativeai
        PyMuPDF
        sentence-transformers
        requests
        SpeechRecognition
        ```

        ### 🎤 Voice Features Setup
        
        1. **ElevenLabs Account**: Sign up at [ElevenLabs.io](https://elevenlabs.io)
        2. **Get API Key**: Go to your profile settings and copy your API key
        3. **Choose Voice**: Browse available voices and copy the voice ID (optional)
        4. **Browser Support**: Voice input uses Streamlit's built-in audio recording
        
        ### 🔊 Voice Features Include:
        - **Voice Input**: Record your questions using Streamlit's audio input
        - **Speech-to-Text**: Automatic transcription using Google Speech Recognition
        - **AI Processing**: Questions sent to Google Gemini for processing
        - **Text-to-Speech**: Responses converted to speech using ElevenLabs
        - **Audio Playback**: Seamless audio responses in the chat interface
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
    Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, <strong>Pinecone</strong>, <strong>Google Gemini</strong>, and <strong>ElevenLabs</strong>
    <br>
    🎤 Voice-enabled AI PDF Chatbot
</div>
""", unsafe_allow_html=True)
