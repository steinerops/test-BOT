import os
import streamlit as st
import tempfile
import logging
import requests
import base64
import time
import json
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

# D-ID Avatar Handler
class AvatarHandler:
    def __init__(self):
        self.api_key = st.secrets.get("DID_API_KEY", "")
        self.base_url = "https://api.d-id.com"
        self.default_presenter_id = "amy-jZaHvguKB7"  # Default D-ID presenter
        
    def get_available_presenters(self):
        """Get list of available presenters from D-ID"""
        if not self.api_key:
            return []
            
        url = f"{self.base_url}/talks/presenters"
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                presenters = response.json().get("presenters", [])
                return [(p["id"], p["name"]) for p in presenters]
            else:
                logger.error(f"D-ID API error getting presenters: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching D-ID presenters: {str(e)}")
            return []
    
    def create_talking_avatar(self, audio_content, text, presenter_id=None):
        """Create a talking avatar video using D-ID API with ElevenLabs audio"""
        if not self.api_key:
            logger.warning("D-ID API key not found")
            return None
            
        if not presenter_id:
            presenter_id = self.default_presenter_id
            
        # First, upload the audio to D-ID
        audio_url = self._upload_audio(audio_content)
        if not audio_url:
            return None
            
        # Create the talking avatar
        url = f"{self.base_url}/talks"
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "script": {
                "type": "audio",
                "audio_url": audio_url,
                "reduce_noise": True,
                "ssml": False
            },
            "presenter_id": presenter_id,
            "background": {
                "color": "#FFFFFF"
            },
            "config": {
                "result_format": "mp4",
                "fluent": True,
                "pad_audio": 0.0,
                "driver_expressions": {
                    "expressions": [
                        {
                            "start_frame": 0,
                            "expression": "neutral",
                            "intensity": 1.0
                        }
                    ]
                }
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 201:
                talk_id = response.json().get("id")
                logger.info(f"Created D-ID talk with ID: {talk_id}")
                
                # Poll for completion
                video_url = self._poll_for_completion(talk_id)
                return video_url
            else:
                logger.error(f"D-ID API error creating talk: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating D-ID talking avatar: {str(e)}")
            return None
    
    def _upload_audio(self, audio_content):
        """Upload audio content to D-ID and return the URL"""
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
                tmp_audio.write(audio_content)
                audio_path = tmp_audio.name
            
            # Upload to D-ID
            url = f"{self.base_url}/audios"
            headers = {
                "Authorization": f"Basic {self.api_key}",
            }
            
            with open(audio_path, 'rb') as audio_file:
                files = {"audio": ("audio.mp3", audio_file, "audio/mpeg")}
                response = requests.post(url, headers=headers, files=files)
            
            # Clean up temp file
            os.unlink(audio_path)
            
            if response.status_code == 201:
                audio_url = response.json().get("url")
                logger.info(f"Uploaded audio to D-ID: {audio_url}")
                return audio_url
            else:
                logger.error(f"D-ID audio upload error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading audio to D-ID: {str(e)}")
            return None
    
    def _poll_for_completion(self, talk_id, max_wait_time=120):
        """Poll D-ID API until video generation is complete"""
        url = f"{self.base_url}/talks/{talk_id}"
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "done":
                        video_url = data.get("result_url")
                        logger.info(f"D-ID video generation completed: {video_url}")
                        return video_url
                    elif status == "error":
                        logger.error(f"D-ID video generation failed: {data.get('error', 'Unknown error')}")
                        return None
                    else:
                        logger.info(f"D-ID video generation status: {status}")
                        time.sleep(5)  # Wait 5 seconds before next poll
                else:
                    logger.error(f"D-ID polling error: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error polling D-ID status: {str(e)}")
                return None
        
        logger.error("D-ID video generation timed out")
        return None

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


def create_audio_player(audio_content, message_id, should_autoplay=False):
    """Create an HTML audio player for the generated audio with proper playback control"""
    audio_base64 = base64.b64encode(audio_content).decode()
    
    audio_html = f"""
    <div id="audio_container_{message_id}">
        <audio id="audio_{message_id}" controls style="width: 100%;">
            <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    <script>
        (function() {{
            const audioElement = document.getElementById('audio_{message_id}');
            
            if (audioElement.hasAttribute('data-processed')) {{
                return;
            }}
            
            audioElement.setAttribute('data-processed', 'true');
            
            audioElement.addEventListener('play', function() {{
                const allAudios = document.querySelectorAll('audio');
                const allVideos = document.querySelectorAll('video');
                
                allAudios.forEach(function(audio) {{
                    if (audio.id !== 'audio_{message_id}' && !audio.paused) {{
                        audio.pause();
                        audio.currentTime = 0;
                    }}
                }});
                
                allVideos.forEach(function(video) {{
                    if (!video.paused) {{
                        video.pause();
                        video.currentTime = 0;
                    }}
                }});
            }});
            
            if ({str(should_autoplay).lower()}) {{
                const allAudios = document.querySelectorAll('audio');
                const allVideos = document.querySelectorAll('video');
                
                allAudios.forEach(function(audio) {{
                    if (!audio.paused) {{
                        audio.pause();
                        audio.currentTime = 0;
                    }}
                }});
                
                allVideos.forEach(function(video) {{
                    if (!video.paused) {{
                        video.pause();
                        video.currentTime = 0;
                    }}
                }});
                
                setTimeout(function() {{
                    audioElement.play().catch(function(error) {{
                        console.log('Autoplay prevented:', error);
                    }});
                }}, 100);
            }}
        }})();
    </script>
    """
    return audio_html

def create_avatar_video_player(video_url, message_id, should_autoplay=False):
    """Create an HTML video player for the avatar video"""
    video_html = f"""
    <div id="avatar_container_{message_id}" style="max-width: 500px; margin: 10px 0;">
        <video id="avatar_{message_id}" controls style="width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video element.
        </video>
    </div>
    <script>
        (function() {{
            const videoElement = document.getElementById('avatar_{message_id}');
            
            if (videoElement.hasAttribute('data-processed')) {{
                return;
            }}
            
            videoElement.setAttribute('data-processed', 'true');
            
            videoElement.addEventListener('play', function() {{
                const allVideos = document.querySelectorAll('video');
                const allAudios = document.querySelectorAll('audio');
                
                allVideos.forEach(function(video) {{
                    if (video.id !== 'avatar_{message_id}' && !video.paused) {{
                        video.pause();
                        video.currentTime = 0;
                    }}
                }});
                
                allAudios.forEach(function(audio) {{
                    if (!audio.paused) {{
                        audio.pause();
                        audio.currentTime = 0;
                    }}
                }});
            }});
            
            if ({str(should_autoplay).lower()}) {{
                const allVideos = document.querySelectorAll('video');
                const allAudios = document.querySelectorAll('audio');
                
                allVideos.forEach(function(video) {{
                    if (!video.paused) {{
                        video.pause();
                        video.currentTime = 0;
                    }}
                }});
                
                allAudios.forEach(function(audio) {{
                    if (!audio.paused) {{
                        audio.pause();
                        audio.currentTime = 0;
                    }}
                }});
                
                setTimeout(function() {{
                    videoElement.play().catch(function(error) {{
                        console.log('Autoplay prevented:', error);
                    }});
                }}, 100);
            }}
        }})();
    </script>
    """
    return video_html

def create_voice_input_interface():
    """Create voice input interface using Streamlit's built-in audio input"""
    st.markdown("### 🎤 Voice Input")
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
        r = sr.Recognizer()
        
        if hasattr(audio_input, 'getvalue'):
            audio_bytes = audio_input.getvalue()
        else:
            audio_bytes = audio_input
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            tmp_audio.write(audio_bytes)
            audio_file_path = tmp_audio.name
        
        try:
            with sr.AudioFile(audio_file_path) as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.record(source)
                
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
st.set_page_config(page_title="AI PDF Chatbot with Avatar", layout="wide")
st.title("🤖 AI PDF Chatbot with Avatar Mode")
st.write("Upload a PDF and chat with it using voice, text, or avatar mode!")

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

if "avatar_handler" not in st.session_state:
    st.session_state.avatar_handler = AvatarHandler()

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "avatar_mode" not in st.session_state:
    st.session_state.avatar_mode = False
if "selected_presenter" not in st.session_state:
    st.session_state.selected_presenter = "amy-jZaHvguKB7"
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "21m00Tcm4TlvDq8ikWAM"
if "last_played_audio" not in st.session_state:
    st.session_state.last_played_audio = None
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0
if "last_autoplay_message_id" not in st.session_state:
    st.session_state.last_autoplay_message_id = -1

# Define the layout
left_col, right_col = st.columns([1, 2])

# Sidebar content
with left_col:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Voice and Avatar settings
    st.header("🎤 Voice & Avatar Settings")
    voice_enabled = st.toggle("Enable Voice Features", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = voice_enabled
    
    # Avatar mode toggle
    avatar_mode = st.toggle("Enable Avatar Mode", value=st.session_state.avatar_mode)
    st.session_state.avatar_mode = avatar_mode
    
    if avatar_mode:
        st.info("🎭 Avatar mode enabled! Responses will be delivered by a talking avatar.")
        
        # Get available presenters
        presenters = st.session_state.avatar_handler.get_available_presenters()
        if presenters:
            presenter_options = {name: id for id, name in presenters}
            selected_presenter_name = st.selectbox(
                "Choose Avatar Presenter",
                options=list(presenter_options.keys()),
                index=0
            )
            st.session_state.selected_presenter = presenter_options[selected_presenter_name]
        else:
            st.warning("Could not load D-ID presenters. Using default avatar.")
            st.session_state.selected_presenter = "amy-jZaHvguKB7"
    
    if voice_enabled or avatar_mode:
        # Get available voices from ElevenLabs
        voices = st.session_state.voice_handler.get_available_voices()
        if voices:
            voice_options = {name: id for id, name in voices}
            selected_voice_name = st.selectbox(
                "Choose Voice",
                options=list(voice_options.keys()),
                index=0 if voice_options else None
            )
            if selected_voice_name:
                st.session_state.selected_voice = voice_options[selected_voice_name]
                st.session_state.voice_handler.voice_id = st.session_state.selected_voice
        else:
            st.warning("Could not load ElevenLabs voices. Using default voice.")

    # Document processing section
    if uploaded_file and st.session_state.chatbot:
        if not st.session_state.document_processed:
            if st.button("Process Document"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Extract text from PDF
                        documents = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                        
                        if documents:
                            # Chunk the text
                            chunks = st.session_state.chatbot.chunk_text(documents)
                            
                            # Create embeddings and store in Pinecone
                            namespace = f"pdf_{uploaded_file.name.replace('.pdf', '').replace(' ', '_')}"
                            vectorstore = st.session_state.chatbot.create_embeddings_and_store(chunks, namespace)
                            
                            # Create conversational chain
                            st.session_state.conversation = st.session_state.chatbot.get_conversational_chain(vectorstore)
                            st.session_state.document_processed = True
                            
                            st.success(f"✅ Document processed successfully! Created {len(chunks)} chunks.")
                        else:
                            st.error("No text could be extracted from the PDF.")
                    except Exception as e:
                        logger.error(f"Error processing document: {str(e)}")
                        st.error(f"Error processing document: {str(e)}")
        else:
            st.success("✅ Document is ready for questions!")
            if st.button("Process New Document"):
                st.session_state.document_processed = False
                st.session_state.conversation = None
                st.session_state.chat_history = []
                st.rerun()

    # Voice input section
    if voice_enabled and has_speech_recognition:
        st.header("🎤 Voice Input")
        audio_input = st.audio_input("Record your question")
        
        if audio_input:
            with st.spinner("Converting speech to text..."):
                voice_query = process_audio_to_text(audio_input)
                if voice_query:
                    st.success(f"Voice input recognized: {voice_query}")
                    # Store the voice query to be used in the main chat
                    st.session_state.voice_query = voice_query

# Main chat interface
with right_col:
    st.header("💬 Chat Interface")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, (query, response, audio_content, video_url) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**AI:** {response}")
            
            # Display audio player if available
            if audio_content and not video_url:  # Only show audio if no video
                should_autoplay = (i == len(st.session_state.chat_history) - 1 and 
                                 st.session_state.last_autoplay_message_id < i)
                audio_html = create_audio_player(audio_content, i, should_autoplay)
                st.components.v1.html(audio_html, height=60)
                if should_autoplay:
                    st.session_state.last_autoplay_message_id = i
            
            # Display avatar video if available
            if video_url:
                should_autoplay = (i == len(st.session_state.chat_history) - 1 and 
                                 st.session_state.last_autoplay_message_id < i)
                video_html = create_avatar_video_player(video_url, i, should_autoplay)
                st.components.v1.html(video_html, height=400)
                if should_autoplay:
                    st.session_state.last_autoplay_message_id = i
            
            st.markdown("---")
    
    # Query input
    query_input = st.text_input("Ask a question about your document:", key="text_query")
    
    # Check for voice query
    voice_query_text = ""
    if hasattr(st.session_state, 'voice_query'):
        voice_query_text = st.session_state.voice_query
        del st.session_state.voice_query  # Clear after use
    
    # Use voice query if available, otherwise use text input
    final_query = voice_query_text if voice_query_text else query_input
    
    # Process query
    if final_query and st.session_state.conversation:
        if st.button("Send", key="send_button") or voice_query_text:
            with st.spinner("Generating response..."):
                try:
                    # Get response from chatbot
                    response, source_docs = st.session_state.chatbot.process_query(
                        st.session_state.conversation, final_query
                    )
                    
                    audio_content = None
                    video_url = None
                    
                    # Generate audio if voice features are enabled
                    if st.session_state.voice_enabled or st.session_state.avatar_mode:
                        with st.spinner("Generating audio..."):
                            audio_content = st.session_state.voice_handler.text_to_speech(response)
                            
                            if not audio_content:
                                st.warning("Could not generate audio for the response.")
                    
                    # Generate avatar video if avatar mode is enabled
                    if st.session_state.avatar_mode and audio_content:
                        with st.spinner("Creating avatar video... This may take 1-2 minutes."):
                            video_url = st.session_state.avatar_handler.create_talking_avatar(
                                audio_content, 
                                response, 
                                st.session_state.selected_presenter
                            )
                            
                            if not video_url:
                                st.warning("Could not generate avatar video. Audio will be used instead.")
                    
                    # Add to chat history
                    st.session_state.chat_history.append((final_query, response, audio_content, video_url))
                    st.session_state.message_counter += 1
                    
                    # Clear input
                    st.session_state.text_query = ""
                    
                    # Rerun to update the interface
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")
    
    elif final_query and not st.session_state.conversation:
        st.warning("Please upload and process a PDF document first!")
    
    # Clear chat history button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.message_counter = 0
            st.session_state.last_autoplay_message_id = -1
            if st.session_state.chatbot:
                st.session_state.chatbot.memory.clear()
            st.rerun()

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. **Upload PDF**: Choose a PDF file and click "Process Document"
2. **Enable Features**: 
   - Toggle "Voice Features" for audio responses
   - Toggle "Avatar Mode" for talking avatar videos with lip-sync
3. **Choose Settings**: Select your preferred avatar and voice
4. **Ask Questions**: Type your question or use voice input (if enabled)
5. **Enjoy**: Get responses in text, audio, or avatar video format!

### 🔧 Required API Keys (in Streamlit secrets):
- `PINECONE_API_KEY`: For vector storage
- `GOOGLE_API_KEY`: For Gemini LLM
- `ELEVENLABS_API_KEY`: For text-to-speech (optional)
- `DID_API_KEY`: For avatar generation (optional)
- `LANGCHAIN_API_KEY`: For tracing (optional)

### 🎭 Avatar Mode Features:
- **Lip-sync**: Perfect synchronization between audio and mouth movements
- **Multiple Avatars**: Choose from various D-ID presenters
- **Voice Selection**: Pick from different ElevenLabs voices
- **Auto-play**: Latest responses play automatically
- **Media Control**: Only one audio/video plays at a time
""")

# Error handling for missing dependencies
if not has_speech_recognition and voice_enabled:
    st.error("Speech recognition is not available. Please install the speech_recognition library.")

# Check for required API keys
required_keys = ["PINECONE_API_KEY", "GOOGLE_API_KEY"]
missing_keys = [key for key in required_keys if key not in st.secrets]

if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}")
    st.info("Please add these keys to your Streamlit secrets.")

# Optional API key warnings
optional_keys = {
    "ELEVENLABS_API_KEY": "Voice features will not work",
    "DID_API_KEY": "Avatar mode will not work"
}

for key, message in optional_keys.items():
    if key not in st.secrets:
        st.warning(f"Optional API key missing ({key}): {message}")

logger.info("Streamlit app initialization complete")
