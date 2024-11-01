import base64
from threading import Lock, Thread
import cv2
from cv2 import VideoCapture, imencode
import openai
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from speech_recognition import Microphone, Recognizer, UnknownValueError
from gtts import gTTS
import io
import pygame
from dotenv import load_dotenv
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


load_dotenv()

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the application."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()
        self.last_captured = None
        self.logger = setup_logger("WebcamStream")

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        self.logger.info("Webcam stream started")
        return self

    def update(self):
        while self.running:
            try:
                ret, frame = self.stream.read()
                if not ret:
                    self.logger.warning("Failed to read frame from webcam")
                    continue
                    
                with self.lock:
                    self.frame = frame
            except Exception as e:
                self.logger.error(f"Error updating frame: {str(e)}")

    def read(self, encode: bool = False):
        try:
            with self.lock:
                frame = self.frame.copy()
            
            if encode:
                ret, buffer = imencode(".jpeg", frame)
                if not ret:
                    raise ValueError("Failed to encode frame")
                return base64.b64encode(buffer)
            return frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return None

    def capture(self):
        """Capture current frame and store it"""
        try:
            self.last_captured = self.read(encode=True)
            if self.last_captured is not None:
                self.logger.info("Frame captured successfully")
            return self.last_captured
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return None

    def get_last_captured(self):
        """Get last captured frame"""
        return self.last_captured

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()
        self.logger.info("Webcam stream stopped")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

class ImageAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 150):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = setup_logger("ImageAnalyzer")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None
    )
    def analyze_image(
        self,
        image_base64: bytes,
        prompt: str = """Dapatkan keyword dari benda itu dan lakukan pencarian pada database"""
    ) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64.decode('utf-8')}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            self.logger.info("Image analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            if hasattr(e, 'response'):
                self.logger.error(f"Response details: {e.response}")
            return None

class DatabaseHandler:
    def __init__(self):
        self.logger = setup_logger("DatabaseHandler")
        self.mongo_uri = os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MongoDB URI not found in environment variables")
            
        try:
            self.client = MongoClient(self.mongo_uri)
            self.collection = self.client['search_db']['search_col']
            
            self.vectorStore = MongoDBAtlasVectorSearch(
                self.collection,
                OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
                index_name='vsearch_index'
            )

            self.SYSTEM_PROMPT = """
            Anda adalah asisten AI bernama Eddy yang ramah dan profesional. 
            Berikan jawaban yang ringkas dan informatif dalam bahasa Indonesia.
            Apabila user menanyakan hal diluar konteks yang tersedia di database, jangan jawab dan rekomendasikan produk apapun, minta user untuk
            melakukan pencarian ulang
            
            Jika anda mendapatkan gambar, CUKUP ANALISA UNTUK KEBUTUHAN QUERY DIDATABASE SAJA! JANGAN PERNAH MENAMPILKAN KEYWORD HASIL ANALISA! gunakan informasi hasil analysa keyword dari gambar tersebut untuk mencari produk yang sesuai di dalam database,
            jika tidak ada produk yang sesuai dengan yang ada di database, jangan tampilkan apapun dan minta user
            melakukan pencarian ulang!
            
            Saat merekomendasikan produk, selalu sertakan:
            1. Nama produk
            2. Harga
            3. Lokasi toko
            4. Link produk

            Konteks percakapan: {context}
            Pertanyaan: {question}
            """

            system_template = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
            self.chat_prompt = ChatPromptTemplate.from_messages([system_template])

            self.llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
                model_name="gpt-4o-mini"
            )

            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorStore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.chat_prompt}
            )
            
            self.logger.info("Database handler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database handler: {str(e)}")
            raise

    def search_products(self, query: str, image_description: Optional[str] = None) -> tuple[str, list]:
        """Search for products based on query and optional image description"""
        try:
            enhanced_query = query
            if image_description:
                enhanced_query = f"{query}. Objek yang terlihat: {image_description}"
                
            self.logger #.info(f"Searching with query: {enhanced_query}")
            result = self.qa.invoke({"question": enhanced_query})
            
            return result["answer"], result.get("source_documents", [])
            
        except Exception as e:
            self.logger.error(f"Error searching products: {str(e)}")
            return "Maaf, terjadi kesalahan dalam pencarian.", []

class VoiceHandler:
    def __init__(self):
        self.logger = setup_logger("VoiceHandler")
        try:
            pygame.mixer.init()
            self.recognizer = Recognizer()
            self.microphone = Microphone()
            self.is_speaking = False
            self.logger.info("Voice handler initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing voice handler: {str(e)}")
            raise
        
    def text_to_speech(self, text: str, lang: str = 'id'):
        """Convert text to speech and play it"""
        try:
            self.is_speaking = True
            tts = gTTS(text=text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            temp_file = "temp_speech.mp3"
            with open(temp_file, "wb") as f:
                f.write(fp.read())
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.is_speaking:
                pygame.time.Clock().tick(10)
            
            os.remove(temp_file)
            self.is_speaking = False
            # self.logger.info("Text-to-speech completed successfully")
            
        except Exception as e:
            self.logger.error(f"TTS Error: {str(e)}")
            self.is_speaking = False

    def stop_speaking(self):
        """Stop current speech playback"""
        if self.is_speaking:
            pygame.mixer.music.stop()
            self.is_speaking = False


class ShoppingAssistant:
    def __init__(self):
        self.logger = setup_logger("ShoppingAssistant")
        self.webcam = None
        self.db_handler = None
        self.voice_handler = None
        self.image_analyzer = None
        self.initialize_components()
        self.SYSTEM_PROMPT = """
        Anda adalah asisten AI bernama Eddy yang ramah dan profesional. 
        Berikan jawaban yang ringkas dan informatif dalam bahasa Indonesia.
        Apabila user menanyakan hal diluar konteks yang tersedia di database, jangan jawab dan rekomendasikan produk apapun, minta user untuk
        melakukan pencarian ulang
        
        Jika anda mendapatkan gambar, CUKUP ANALISA UNTUK KEBUTUHAN QUERY DIDATABASE SAJA! JANGAN PERNAH MENAMPILKAN KEYWORD HASIL ANALISA! gunakan informasi hasil analysa keyword dari gambar tersebut untuk mencari produk yang sesuai di dalam database,
        jika tidak ada produk yang sesuai dengan yang ada di database, jangan tampilkan apapun dan minta user
        melakukan pencarian ulang!
        
        Saat merekomendasikan produk, selalu sertakan:
        1. Nama produk
        2. Harga
        3. Lokasi toko
        4. Link produk

        Konteks percakapan: {context}
        Pertanyaan: {question}
        """

        system_template = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        self.chat_prompt = ChatPromptTemplate.from_messages([system_template])

    def initialize_components(self):
        """Initialize all components with error handling"""
        try:
            self.webcam = WebcamStream().start()
            self.db_handler = DatabaseHandler()
            self.voice_handler = VoiceHandler()
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            self.image_analyzer = ImageAnalyzer(api_key)
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def format_response(self, answer: str, docs: list) -> tuple[str, str]:
        """
        Format the response with product recommendations and return both speech and display text
        Returns:
            tuple: (speech_text, display_text)
        """
        try:
            # Extract first sentence for speech
            intro_sentence = answer.split(':')[0] + '.'
            
            # Format full response for display
            display_response = f"\n\n{answer}\n\n"
            if docs:
                display_response += "Produk yang ditemukan:\n"
                for idx, doc in enumerate(docs, 1):
                    metadata = doc.metadata
                    display_response += f"{idx}. {metadata.get('NamaProduk', 'N/A')}\n"
                    display_response += f"   Harga: {metadata.get('HargaProduk', 'N/A')}\n"
                    display_response += f"   Lokasi: {metadata.get('LokasiToko', 'N/A')}\n"
                    display_response += f"   Link: {metadata.get('LinkProduk', 'N/A')}\n\n"
            
            return intro_sentence, display_response
                
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return "Maaf, terjadi kesalahan.", "Maaf, terjadi kesalahan dalam memformat respons, mohon diulangi lagi."


    def should_capture_image(self, text: str) -> bool:
        """Check if the query requires image analysis"""
        if not text:
            return False
        trigger_phrases = [
            "lihat", "coba lihat", "benda ini", "produk ini",
            "barang ini", "seperti ini", "mirip ini"
        ]
        return any(phrase in text.lower() for phrase in trigger_phrases)

    def audio_callback(self, recognizer, audio):
        """Process audio input and generate response"""
        try:
            text = recognizer.recognize_whisper(audio, model="base", language="indonesian")
            self.logger.info(f"User said: {text}")
            
            image_description = None
            if self.should_capture_image(text):
                image = self.webcam.capture()
                if image is not None:
                    self.logger.info("Analyzing image...")
                    image_description = self.image_analyzer.analyze_image(image)
                    if image_description:
                        self.logger.info("Image analysis completed")
            
            answer, docs = self.db_handler.search_products(text, image_description)
            speech_text, display_text = self.format_response(answer, docs)
            
            # Display full response
            print(display_text)
            
            # Speak only the introductory sentence
            self.voice_handler.text_to_speech(speech_text)
            
        except UnknownValueError:
            self.logger.warning("Could not understand audio")
            self.voice_handler.text_to_speech("Maaf, saya tidak dapat memahami suara Anda. Mohon ulangi.")
        except Exception as e:
            self.logger.error(f"Error in audio callback: {str(e)}")
            self.voice_handler.text_to_speech("Maaf, terjadi kesalahan. Mohon coba lagi.")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.webcam:
                self.webcam.stop()
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def run(self):
        """Run the shopping assistant with error handling"""
        stop_listening = None
        try:
            with self.voice_handler.microphone as source:
                self.voice_handler.recognizer.adjust_for_ambient_noise(source)
            
            stop_listening = self.voice_handler.recognizer.listen_in_background(
                self.voice_handler.microphone, 
                self.audio_callback
            )
            self.logger.info("Shopping assistant started successfully")
            
            while True:
                frame = self.webcam.read()
                if frame is None:
                    self.logger.warning("Failed to read frame from webcam")
                    continue
                    
                if self.webcam.get_last_captured() is not None:
                    cv2.putText(frame, "Image Captured!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Webcam", frame)
                key = cv2.waitKey(1)
                if key in [27, ord("q")]:
                    break
                
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            if stop_listening:
                stop_listening(wait_for_stop=False)
            self.cleanup()

if __name__ == "__main__":
    try:
        assistant = ShoppingAssistant()
        assistant.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")