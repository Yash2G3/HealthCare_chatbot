# Healthcare Assistant Chatbot

A Streamlit-based healthcare assistant chatbot powered by Groq's LLM API. This application offers multiple modes of interaction:

- 🩺 **Health Assessment**: Complete diagnostic questionnaire to receive a basic health assessment and recommendations
- 🏥 **Hospital Navigation**: Get directions to various departments within the hospital
- 💬 **Friendly Conversation**: Chat with an empathetic assistant for emotional support
- 🔍 **Medical Inquiry**: Ask specific healthcare questions with medically informed responses

## Features

- Interactive chat interface with multiple interaction modes
- Voice input and text-to-speech capabilities
- Progressive health assessment with detailed reports
- Responsive and user-friendly interface

## Setup and Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Groq API key: `GROQ_API_KEY=your_api_key_here`
4. Run the FastAPI backend: `uvicorn app:app --reload`
5. Run the Streamlit frontend: `streamlit run streamlit_app.py`

## Technologies Used

- Streamlit: Frontend UI framework
- FastAPI: Backend API server
- Groq API: LLM provider for natural language processing
- gTTS: Google Text-to-Speech for voice responses
- SpeechRecognition: For voice input processing

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `app.py`: FastAPI backend server
- `hospital_layout.txt`: Information about hospital layout for navigation
- `requirements.txt`: Project dependencies