import streamlit as st
# import requests # No longer needed for core chat functionality
import json
import uuid
import base64
from io import BytesIO
import datetime
import os
import speech_recognition as sr
from gtts import gTTS
import pyaudio # Import to check installation
from dotenv import load_dotenv
from groq import Groq, GroqError # Use Groq client directly
import time # Import time for potential delays

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Microphone Check ---
MICROPHONE_AVAILABLE = False
MICROPHONE_ERROR_MSG = ""
try:
    # Check if PyAudio is installed and usable
    p = pyaudio.PyAudio()
    p.terminate()
    # Check if SpeechRecognition can list microphones
    mic_list = sr.Microphone.list_microphone_names()
    if not mic_list:
        MICROPHONE_ERROR_MSG = "No microphones found by SpeechRecognition. Please check your system's audio input devices."
    else:
        MICROPHONE_AVAILABLE = True
        print("Microphones found:", mic_list) # Optional: Log found mics
except NameError:
    MICROPHONE_ERROR_MSG = "PyAudio library not found or installed correctly. Please run: pip install pyaudio"
except ImportError: # Catch ImportError specifically for pyaudio
    MICROPHONE_ERROR_MSG = "PyAudio library not found or installed correctly. Please run: pip install pyaudio"
except Exception as e:
    MICROPHONE_ERROR_MSG = f"Error initializing audio input: {e}. Ensure microphone is connected and permissions are granted."
    # On macOS, you might need to grant terminal/IDE microphone access in System Settings > Privacy & Security > Microphone.

# --- Hospital Layout ---
try:
    with open("hospital_layout.txt", "r") as f:
        HOSPITAL_LAYOUT = f.read()
except FileNotFoundError:
    HOSPITAL_LAYOUT = "Hospital layout information is currently unavailable."
    st.warning("Warning: hospital_layout.txt not found. Navigation features will be limited.")

# --- Groq Client Initialization ---
def get_groq_client():
    """Initializes and returns the Groq client."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
        st.stop() # Stop execution if key is missing
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Test the client with a simple call (optional, but good for debugging)
        # client.chat.completions.create(messages=[{"role": "user", "content": "test"}], model="llama3-8b-8192", max_tokens=10)
        return client
    except GroqError as e:
        st.error(f"Error initializing Groq client: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Groq client initialization: {e}")
        st.stop()

# Initialize client globally or pass it around
# Global initialization might be simpler for Streamlit's execution model
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = get_groq_client()

# --- System Prompts ---
HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare chatbot designed to provide helpful and comforting information to patients.
You must respond in the following JSON format only:
{
    "chatbot_answer": "<your detailed and empathetic healthcare response>",
    "category": "<one of: require_human_intervention, require_medicine, personal_assistance_needed, hospital_info>"
}

Guidelines for responses:
- chatbot_answer: Provide clear, compassionate, and helpful healthcare information. Avoid giving specific medical diagnoses or treatment plans. Focus on general information, symptom management advice (like rest, hydration), and when to seek professional help.
- category must be exactly one of these four options:
  * require_human_intervention: For serious medical concerns suggesting immediate professional attention (e.g., severe chest pain, difficulty breathing, high fever with confusion).
  * require_medicine: For conditions that might benefit from over-the-counter medication or pharmacy advice (e.g., mild headache, common cold symptoms).
  * personal_assistance_needed: For situations requiring caregiver or personal support (e.g., help with mobility, loneliness).
  * hospital_info: For general hospital/clinic information queries, appointment scheduling, or non-urgent health questions.

The response must be valid JSON. Do not include any text outside the JSON structure."""

CONVERSATION_SYSTEM_PROMPT = """You are a friendly and empathetic healthcare chatbot designed to provide comfort and
companionship to patients who might be feeling low or anxious. Engage in a warm, supportive conversation with the patient.
Your responses should be brief (1-3 sentences), empathetic, and encouraging without giving specific medical advice.
Focus on being a good listener and asking thoughtful follow-up questions to show you care."""

NAVIGATION_SYSTEM_PROMPT = f"""You are a hospital navigation assistant. Your goal is to help patients and visitors
find their way around the hospital based on the hospital layout information provided below. Your responses should be clear,
concise, and helpful. When giving directions, provide step-by-step instructions using landmarks and room numbers if available.

Hospital Layout Information:
---
{HOSPITAL_LAYOUT}
---

Always respond in a friendly tone and ask if they need further clarification on the directions."""

# Note: DIAGNOSIS_SYSTEM_PROMPT is not used as diagnosis is handled locally.

# Set up page configuration
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Diagnosis Specific Functions ----------

DIAGNOSIS_QUESTIONS = {
    1: {"text": "Please tell me your age:", "field": "age", "category": "patient_info"},
    2: {"text": "What is your gender? (Male/Female/Other):", "field": "gender", "category": "patient_info"},
    3: {"text": "What is your weight in kg?", "field": "weight", "category": "patient_info"},
    4: {"text": "What is your height in cm?", "field": "height", "category": "patient_info"},
    5: {"text": "What is your blood group? (Optional)", "field": "blood_group", "category": "patient_info"},
    6: {"text": "What symptoms are you experiencing? Please list the main ones.", "field": "main_symptoms", "category": "symptoms"},
    7: {"text": "How long have you had these symptoms? (e.g., 2 days, 1 week)", "field": "duration", "category": "symptoms"},
    8: {"text": "On a scale of 1 (mild) to 10 (severe), how severe is your discomfort?", "field": "severity", "category": "symptoms"},
    9: {"text": "Do you have any existing medical conditions? (e.g., Diabetes, Asthma, None)", "field": "conditions", "category": "medical_history"},
    10: {"text": "Are you currently taking any medications? Please list them or say None.", "field": "medications", "category": "medical_history"}
}

def validate_diagnosis_answer(question_num, answer):
    """Validate user input based on question type"""
    answer = str(answer).strip()

    if not answer:
        return "Please provide an answer."

    q_info = DIAGNOSIS_QUESTIONS.get(question_num)
    if not q_info:
        return None # Should not happen

    field = q_info["field"]

    if field == "age":
        if not answer.isdigit() or not 0 < int(answer) <= 120:
            return "Please enter a valid age (e.g., 35)."
    elif field == "gender":
        if answer.lower() not in ['male', 'female', 'other']:
            return "Please enter Male, Female, or Other."
    elif field == "weight":
        try:
            weight = float(answer)
            if not 1 <= weight <= 500:
                return "Please enter a valid weight in kg (e.g., 70)."
        except ValueError:
            return "Please enter a valid number for weight (e.g., 70)."
    elif field == "height":
        try:
            height = float(answer)
            if not 30 <= height <= 300:
                return "Please enter a valid height in cm (e.g., 175)."
        except ValueError:
            return "Please enter a valid number for height (e.g., 175)."
    elif field == "severity":
        if not answer.isdigit() or not 1 <= int(answer) <= 10:
            return "Please enter a number between 1 and 10."
    # No specific validation for symptoms, duration, conditions, medications, blood_group beyond non-empty

    return None

def analyze_diagnosis_results():
    """Analyze collected diagnosis data locally and determine if doctor is needed"""
    try:
        print("Analyzing diagnosis results locally...")
        ds = st.session_state.diagnosis_state
        print(f"Current diagnosis state: {ds}")

        # Extract severity with careful type handling
        severity_val = ds.get("symptoms", {}).get("severity", "0")
        severity = 0
        try:
            if isinstance(severity_val, str) and severity_val.isdigit():
                severity = int(severity_val)
            elif isinstance(severity_val, (int, float)):
                 severity = int(severity_val)
            # Ensure severity is within bounds
            severity = max(1, min(10, severity))
        except (ValueError, TypeError):
            severity = 0 # Default to 0 if conversion fails

        print(f"Severity determined to be: {severity}")

        # Make assessment based on severity
        # Adjusted threshold from previous version
        call_doctor = severity >= 8 # Use threshold from old logic
        if severity >= 8:
            condition = "severe"
        elif severity >= 5:
            condition = "moderate"
        elif severity >= 1:
            condition = "mild"
        else:
            condition = "undetermined" # If severity couldn't be parsed

        # Update diagnosis state with assessment
        ds.update({
            "call_doctor": call_doctor,
            "patient_condition": condition,
            "analysis_complete": True
        })

        print(f"Local analysis complete. Condition: {condition}, Call doctor: {call_doctor}")
        return f"Analysis complete. Your condition appears to be {condition}."

    except Exception as e:
        print(f"Error in local analysis: {str(e)}")
        st.error(f"Analysis error: {str(e)}")
        # Attempt to recover or set default state
        st.session_state.diagnosis_state.update({
            "call_doctor": False,
            "patient_condition": "error",
            "analysis_complete": True # Mark as complete even on error to show results
        })
        return f"Error analyzing results: {str(e)}"

def generate_recommendations(diagnosis_state):
    """Generate recommendations based on local diagnosis state"""
    if diagnosis_state.get('call_doctor', False): # Use .get for safety
        return "URGENT: Please seek immediate medical attention." # Match old logic

    severity = 0
    try:
        severity_val = diagnosis_state.get('symptoms', {}).get('severity', '0')
        if isinstance(severity_val, str) and severity_val.isdigit():
            severity = int(severity_val)
        elif isinstance(severity_val, (int, float)):
            severity = int(severity_val)
        severity = max(1, min(10, severity))
    except (ValueError, TypeError):
        severity = 0

    recommendations = []
    # Match thresholds from old logic
    if severity >= 7:
         recommendations.append("- Schedule an appointment with a doctor soon")
    elif severity >= 4:
        recommendations.append("- Monitor symptoms closely")
        recommendations.append("- If symptoms worsen, consult a healthcare provider")
    else: # Includes severity 1-3 and 0 (undetermined)
        recommendations.append("- Rest and monitor symptoms")
        recommendations.append("- Practice self-care measures")

    return "\n".join(recommendations)


def generate_diagnosis_report(diagnosis_state):
    """Generate and save diagnosis report"""
    # Helper function to safely convert lists/values to string
    def safe_format(value, default="Not provided"):
        if value is None or value == "":
            return default
        if isinstance(value, list):
            if not value: return default
            # Ensure all items are strings before joining
            return ', '.join(str(item) for item in value if item)
        return str(value)

    ds = diagnosis_state # shorter alias

    report = f"""
MEDICAL CONSULTATION REPORT (Generated by AI Assistant)
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {ds.get('session_id', 'N/A')}

--- PATIENT INFORMATION ---
Age: {safe_format(ds.get('patient_info', {}).get('age'))}
Gender: {safe_format(ds.get('patient_info', {}).get('gender'))}
Weight: {safe_format(ds.get('patient_info', {}).get('weight'))} kg
Height: {safe_format(ds.get('patient_info', {}).get('height'))} cm
Blood Group: {safe_format(ds.get('patient_info', {}).get('blood_group'))}

--- SYMPTOMS AND CONDITION ---
Main Symptoms: {safe_format(ds.get('symptoms', {}).get('main_symptoms'))}
Duration: {safe_format(ds.get('symptoms', {}).get('duration'))}
Severity (1-10): {safe_format(ds.get('symptoms', {}).get('severity'))}

--- MEDICAL HISTORY ---
Existing Conditions: {safe_format(ds.get('medical_history', {}).get('conditions'))}
Current Medications: {safe_format(ds.get('medical_history', {}).get('medications'))}

--- AI ASSESSMENT ---
Assessed Condition: {safe_format(ds.get('patient_condition', 'PENDING'), 'Pending').upper()}
Requires Prompt Medical Attention: {'Yes' if ds.get('call_doctor', False) else 'No'}

--- AI RECOMMENDATIONS ---
{generate_recommendations(ds)}

--- DISCLAIMER ---
This report is generated by an AI assistant based on the information provided.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
"""

    # Create output directory if it doesn't exist
    output_dir = 'diagnosis_reports'
    os.makedirs(output_dir, exist_ok=True)

    # Save report
    filename = f"{output_dir}/report_{ds.get('session_id', 'unknown_session')}.txt"
    try:
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Error saving report {filename}: {e}")
        st.error(f"Could not save report: {e}")
        return report, None # Return report content even if saving fails

    return report, filename


# ---------- Session State Initialization ----------
# Ensure all necessary keys are initialized
default_diagnosis_state = {
    "session_id": str(uuid.uuid4()),
    "current_question": 1,
    "patient_info": {},
    "symptoms": {},
    "medical_history": {},
    "call_doctor": False,
    "patient_condition": "pending",
    "analysis_complete": False,
    "show_results": False,
    "summary": "" # To store the final summary for TTS
}

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "conversation" # Default mode
if 'diagnosis_state' not in st.session_state:
    st.session_state.diagnosis_state = default_diagnosis_state.copy()
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'user_speech' not in st.session_state:
    st.session_state.user_speech = ""
if 'audio_to_play' not in st.session_state: # New state variable for TTS
    st.session_state.audio_to_play = None
if 'audio_key' not in st.session_state: # New state variable for TTS key
    st.session_state.audio_key = None

# ---------- UI Configuration (Reverted to Old Style) ----------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stChatMessage { background-color: #1e2530; border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stTextInput>div>div>input { border-radius: 25px; background-color: #262730; color: white; }
    /* Target only the sidebar microphone button for circle style */
    .stSidebar .stButton>button {
        border-radius: 50% !important; /* Use !important to override potential conflicts */
        height: 50px !important;
        width: 50px !important;
        font-size: 24px !important;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0; /* Remove default padding */
        margin: auto; /* Center button if container width is larger */
    }
    .diagnosis-progress { padding: 1rem; border-radius: 10px; background: #1e2530; margin-bottom: 1rem; } /* Added margin-bottom */
    .stProgress > div > div > div > div { background-color: #4CAF50; } /* Progress bar color (example: green) */
    h1, h2, h3, h4, h5, h6 { color: #fafafa; } /* Headers color */
    .stRadio > label { margin-bottom: 0.5rem; font-weight: 500; } /* Sidebar radio label */
     .stSpinner > div { /* Center spinner */
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- UI Components ----------
def show_diagnosis_progress():
    """Display diagnosis progress bar and current question."""
    ds = st.session_state.diagnosis_state
    current_q = ds.get("current_question", 1)
    total_q = len(DIAGNOSIS_QUESTIONS)

    # Only show progress if diagnosis hasn't completed analysis yet
    if not ds.get("analysis_complete", False) and 1 <= current_q <= total_q:
        # Use the class for styling
        with st.container(): # Removed border=True
             st.markdown('<div class="diagnosis-progress">', unsafe_allow_html=True) # Add class div
             st.markdown("##### Health Assessment Progress") # Use H5 for smaller heading
             progress = (current_q -1) / total_q # Progress based on completed questions
             st.progress(progress, text=f"Question {current_q} of {total_q}")
             # Display the current question text clearly
             st.caption(f"Current Question: {DIAGNOSIS_QUESTIONS[current_q]['text']}") # Use caption
             st.markdown('</div>', unsafe_allow_html=True) # Close class div

def show_diagnosis_results():
    """Display final assessment results and report download."""
    ds = st.session_state.diagnosis_state

    # Ensure analysis is complete before showing results
    if not ds.get("analysis_complete", False):
        print("Attempted to show results before analysis was complete.")
        return

    print("Displaying diagnosis results area...")
    st.divider()
    st.subheader("Assessment Result")

    # Generate report content (don't save again if already generated)
    report_content, _ = generate_diagnosis_report(ds) # We only need content here

    if ds.get("call_doctor", False):
        st.error("⚠️ Immediate Medical Attention Recommended")
        # Match old message
        st.write("A healthcare professional will contact you shortly.")
    else:
        st.success("✅ Assessment Complete")
        # No extra message in old version

    # Display report in expandable section
    with st.expander("View Detailed Report", expanded=True): # Match old version (expanded=True)
        # Use st.text_area instead of st.text to allow scrolling for long reports
        st.text_area("Report Details", report_content, height=300, key=f"report_text_{ds['session_id']}")
        st.download_button(
            label="Download Report", # Match old label
            data=report_content,
            file_name=f"medical_report_{ds['session_id']}.txt",
            mime="text/plain",
            key=f"download_report_{ds['session_id']}"
        )

    if st.button("Start New Assessment", key=f"new_assessment_{ds['session_id']}"):
        print("Starting new assessment...")
        # Reset diagnosis state completely
        st.session_state.diagnosis_state = default_diagnosis_state.copy()
        st.session_state.diagnosis_state["session_id"] = str(uuid.uuid4()) # New session ID
        # Clear chat messages
        st.session_state.messages = [] # Clear all messages like old version
        st.session_state.audio_to_play = None # Clear pending audio
        st.session_state.audio_key = None
        st.rerun()

# ---------- Conversation Context ----------
def get_conversation_context(max_history=5):
    """Get the recent conversation context for the LLM."""
    context = []
    # Include relevant past messages, limiting history length
    for msg in st.session_state.messages[-max_history:]:
        # Ensure role and content are present
        if "role" in msg and "content" in msg:
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    return context

# ---------- Input Handling (Unified & Fixed) ----------
def handle_user_input(user_input):
    """Process user input based on the current mode. Returns the bot response."""
    mode = st.session_state.current_mode
    model = st.session_state.selected_model # Get selected model
    bot_response = None # Initialize bot_response

    # --- Diagnosis Mode (Local Logic) ---
    if mode == "diagnosis":
        ds = st.session_state.diagnosis_state
        current_q = ds["current_question"]
        total_q = len(DIAGNOSIS_QUESTIONS)

        # Add user message to chat history (mark with mode) - This is now done before calling this function

        # Validate input first
        validation_msg = validate_diagnosis_answer(current_q, user_input)
        if validation_msg:
            bot_response = f"{validation_msg}\n\n{DIAGNOSIS_QUESTIONS[current_q]['text']}"
            # Add assistant message to state immediately for validation errors
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response # Return validation error to user

        # --- Store Valid Answer ---
        try:
            question_info = DIAGNOSIS_QUESTIONS[current_q]
            category = question_info["category"]
            field = question_info["field"]

            # Ensure category exists
            if category not in ds:
                ds[category] = {}

            # Store answer (handle list fields specifically)
            if field in ["main_symptoms", "conditions", "medications"]:
                 # Store as list containing the single string input
                 ds[category][field] = [str(user_input)]
            else:
                 # Store scalar fields as string
                 ds[category][field] = str(user_input)
            print(f"Stored Q{current_q} ({field}): {user_input}")

            # --- Move to Next Question or Analyze ---
            next_q = current_q + 1
            if next_q > total_q:
                # --- Analysis Phase ---
                print("All questions answered. Running local analysis...")
                ds["current_question"] = next_q # Mark as past last question
                analysis_summary_msg = analyze_diagnosis_results() # Run local analysis

                # Generate the final summary message for chat and TTS (Match old format)
                condition = ds.get("patient_condition", "undefined").upper()
                call_doctor = ds.get("call_doctor", False)
                recommendations = generate_recommendations(ds) # Use the recommendation function

                summary_message = f"""
Thank you for providing all the information. Here's my assessment:

ASSESSMENT:
---------
Your condition appears to be: {condition}
Need immediate medical attention: {"Yes" if call_doctor else "No"}

RECOMMENDATIONS:
-------------
{recommendations}

A detailed report has been generated for your records.
                """
                bot_response = summary_message
                ds["show_results"] = True # Flag to display the results section
                ds["summary"] = summary_message # Store for potential TTS

            else:
                # --- Ask Next Question ---
                ds["current_question"] = next_q
                bot_response = DIAGNOSIS_QUESTIONS[next_q]["text"]

            # Add the assistant message (next question or summary) to state
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response

        except Exception as e:
            st.error(f"An error occurred during diagnosis step {current_q}: {e}")
            bot_response = f"Sorry, an error occurred. Let's try that question again: {DIAGNOSIS_QUESTIONS[current_q]['text']}"
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response

    # --- Other Modes (Groq API Call - FIXED) ---
    else:
        # User message is added before calling this function

        # Select system prompt based on mode
        if mode == "conversation":
            system_prompt = CONVERSATION_SYSTEM_PROMPT
        elif mode == "navigation":
            system_prompt = NAVIGATION_SYSTEM_PROMPT
        elif mode == "healthcare":
            system_prompt = HEALTHCARE_SYSTEM_PROMPT
        else: # Should not happen
            st.error(f"Invalid mode '{mode}' encountered.")
            bot_response = "Sorry, I encountered an internal error (invalid mode)."
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response

        # Prepare messages for Groq
        groq_messages = [
            {"role": "system", "content": system_prompt}
        ]
        # Add conversation history (limit length)
        groq_messages.extend(get_conversation_context(max_history=5))
        # Ensure the last message is the user's current input (already added to history)


        print(f"--- Calling Groq API (Mode: {mode}, Model: {model}) ---")
        # print(f"Messages: {json.dumps(groq_messages, indent=2)}") # Optional debug

        try:
            client = st.session_state.groq_client
            chat_completion = client.chat.completions.create(
                messages=groq_messages,
                model=model,
                temperature=0.2, # Keep temperature low for consistency
                max_tokens=500, # Adjust as needed
            )
            ai_content = chat_completion.choices[0].message.content
            print(f"Groq Raw Response: {ai_content[:100]}...") # Log snippet

            # --- Process Groq Response ---
            if mode == "healthcare":
                try:
                    # Attempt to parse the JSON response
                    # Handle potential markdown code blocks
                    cleaned_content = ai_content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]

                    response_dict = json.loads(cleaned_content.strip())
                    bot_response = response_dict.get("chatbot_answer", "Sorry, I couldn't formulate a response.")
                    category = response_dict.get("category", "hospital_info") # Default category
                    print(f"Healthcare category: {category}")
                except json.JSONDecodeError:
                    print("Failed to parse JSON response from healthcare mode. Using raw response.")
                    # Fallback: return the raw response or a generic message
                    bot_response = ai_content # Use raw content as fallback
                except Exception as e:
                     print(f"Error processing healthcare response: {e}")
                     bot_response = "Sorry, I encountered an error processing the healthcare information."

            else: # Conversation or Navigation mode
                bot_response = ai_content # Use the text response directly

            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response

        except GroqError as e:
            st.error(f"Groq API Error: {e}")
            bot_response = f"Sorry, I couldn't connect to the AI assistant ({e.type}). Please check your API key and network connection."
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response
        except Exception as e:
            st.error(f"An unexpected error occurred while contacting the AI: {e}")
            bot_response = "Sorry, an unexpected error occurred. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "mode": mode})
            return bot_response

# ---------- Function to Generate and Store TTS Audio ----------
def generate_and_store_tts(text_to_speak):
    """Generates TTS audio bytes and stores them in session state."""
    if not text_to_speak:
        st.session_state.audio_to_play = None
        st.session_state.audio_key = None
        return

    try:
        print(f"Generating TTS for: {text_to_speak[:50]}...")
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        st.session_state.audio_to_play = audio_bytes
        st.session_state.audio_key = f"tts_{uuid.uuid4()}" # Generate a unique key
        print("TTS audio generated and stored in session state.")
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        st.session_state.audio_to_play = None
        st.session_state.audio_key = None

# ---------- Main App Layout ----------
st.title("🏥 Hospital Healthcare Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("Assistant Mode")
    mode_options = ["Friendly Conversation", "Hospital Navigation", "Health Assessment", "Medical Inquiry"]
    # Get current index based on session state
    current_mode_key = st.session_state.current_mode
    mode_map_inverse = {v: k for k, v in {
        "Friendly Conversation": "conversation",
        "Hospital Navigation": "navigation",
        "Health Assessment": "diagnosis",
        "Medical Inquiry": "healthcare"
    }.items()}
    try:
        current_index = mode_options.index(mode_map_inverse.get(current_mode_key, "Friendly Conversation"))
    except ValueError:
        current_index = 0 # Default if mode somehow gets corrupted

    selected_mode_display = st.radio(
        "Select mode", # Match old label
        mode_options,
        index=current_index, # Set index based on current state
        key="mode_radio",
        on_change=lambda: st.session_state.update(audio_to_play=None, audio_key=None) # Clear audio on mode change
    )
    # Map display name to internal mode key
    mode_map = {
        "Friendly Conversation": "conversation",
        "Hospital Navigation": "navigation",
        "Health Assessment": "diagnosis",
        "Medical Inquiry": "healthcare"
    }
    new_mode = mode_map[selected_mode_display]

    # Reset diagnosis state if switching *away* from diagnosis mode
    if st.session_state.current_mode == "diagnosis" and new_mode != "diagnosis":
        print("Switching away from diagnosis mode, resetting state.")
        st.session_state.diagnosis_state = default_diagnosis_state.copy()
        st.session_state.diagnosis_state["session_id"] = str(uuid.uuid4())
        # Clear all messages when switching away from diagnosis
        st.session_state.messages = []
        st.session_state.audio_to_play = None # Clear pending audio
        st.session_state.audio_key = None

    # Reset messages and state if switching *to* diagnosis mode
    elif st.session_state.current_mode != "diagnosis" and new_mode == "diagnosis":
         print("Switching to diagnosis mode, resetting state and messages.")
         st.session_state.diagnosis_state = default_diagnosis_state.copy()
         st.session_state.diagnosis_state["session_id"] = str(uuid.uuid4())
         st.session_state.messages = [] # Clear messages for a fresh start
         st.session_state.audio_to_play = None # Clear pending audio
         st.session_state.audio_key = None

    st.session_state.current_mode = new_mode

    st.divider()
    st.subheader("Audio Controls")
    # Match old default values (True)
    text_to_speech = st.checkbox("Enable Text-to-Speech", value=True, key="tts_checkbox")
    speech_to_text_enabled = st.checkbox( # Renamed variable for clarity
        "Enable Speech-to-Text",
        value=True, # Match old default
        disabled=not MICROPHONE_AVAILABLE,
        help=MICROPHONE_ERROR_MSG if not MICROPHONE_AVAILABLE else "Requires a working microphone", # Match old help text
        key="stt_checkbox"
    )
    if not MICROPHONE_AVAILABLE:
        st.warning(f"🎤 {MICROPHONE_ERROR_MSG}")

    # --- Microphone Button Moved Here ---
    if speech_to_text_enabled and MICROPHONE_AVAILABLE:
        # Use columns to center the button better if needed, or rely on CSS
        # col1, col2, col3 = st.columns([1,1,1])
        # with col2:
        if not st.session_state.get('listening', False):
            if st.button("🎤", key="mic_button_sidebar", help="Click to speak"): # Use icon only
                st.session_state.listening = True
                st.session_state.user_speech = "" # Clear previous speech
                st.session_state.audio_to_play = None # Clear pending audio playback
                st.session_state.audio_key = None
                # Use toast for brief feedback without disrupting layout
                st.toast("Listening...")
                st.rerun() # Rerun to start listening process in main body
        else:
            # Show a "Listening" indicator (can be a disabled button or text)
            st.button("🔴", disabled=True, key="mic_listening_indicator", help="Listening...") # Use icon only


    st.divider()
    st.subheader("Model Selection")
    # Match old model list and label
    model = st.selectbox(
        "Select model from Groq Cloud",
        ["llama3-70b-8192", "llama-3.3-70b-versatile"], # Match old list
        index=0, # Default to Llama 3 70b
        key="model_select"
    )
    st.session_state.selected_model = model # Store selected model in session state

    # No info box in old version's sidebar


# --- Main Chat Area ---

# Display Diagnosis Progress Bar if in diagnosis mode and not yet complete
if st.session_state.current_mode == "diagnosis":
    show_diagnosis_progress()

# Chat Interface
st.subheader("Conversation") # Keep this header
# Use a container, but don't set height to allow natural expansion
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # --- Play TTS Audio If Available ---
    # Check if there's audio data and a key stored in session state
    if st.session_state.get('audio_to_play') and st.session_state.get('audio_key'):
        print(f"Attempting to play audio with key: {st.session_state.audio_key}")
        # Remove the unsupported 'key' argument
        st.audio(st.session_state.audio_to_play, format='audio/mp3', start_time=0, autoplay=True)
        # Clear the audio data and key from state AFTER rendering the widget
        # This prevents it from trying to play again on subsequent non-related reruns
        st.session_state.audio_to_play = None
        st.session_state.audio_key = None


# Display Diagnosis Results Area if needed (triggered by state)
# This ensures it appears *below* the chat container but *above* the input
if st.session_state.current_mode == "diagnosis":
    ds = st.session_state.diagnosis_state
    # Check both flags as before
    if ds.get("show_results", False) and ds.get("analysis_complete", False):
         show_diagnosis_results()


# --- Handle Speech Recognition (if listening state is active) ---
# Moved this section before the input area processing
user_input_speech = ""
if st.session_state.get('listening', False):
    # Display listening status more prominently in the main area
    status_placeholder = st.empty()
    status_placeholder.info("🎙️ Listening... Please speak clearly.")
    try:
        r = sr.Recognizer()
        # Adjust microphone settings if needed (e.g., device_index)
        with sr.Microphone() as source:
            # Optional: Adjust for ambient noise - can add latency but improve accuracy
            # print("Adjusting for ambient noise...")
            # r.adjust_for_ambient_noise(source, duration=0.5)
            # print("Adjustment complete.")
            print("DEBUG: Microphone source opened. Listening...") # Debug print
            # Listen with timeouts
            # Increased timeout slightly
            audio = r.listen(source, timeout=7, phrase_time_limit=15)
            print("DEBUG: Audio captured. Processing...") # Debug print
            status_placeholder.info("🧠 Processing speech...") # Update status
            # Use Google Speech Recognition (requires internet)
            user_input_speech = r.recognize_google(audio)
            st.session_state.user_speech = user_input_speech # Store recognized speech
            print(f"DEBUG: Speech recognized: {user_input_speech}") # Debug print
            st.toast("✅ Speech recognized!") # Use toast for success
    except sr.WaitTimeoutError:
        st.toast("⚠️ No speech detected within the time limit.")
        st.session_state.user_speech = "" # Clear speech on error
    except sr.UnknownValueError:
        st.toast("🤔 Could not understand audio. Please try again.")
        st.session_state.user_speech = "" # Clear speech on error
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        st.session_state.user_speech = "" # Clear speech on error
    except Exception as e:
        st.error(f"An error occurred during speech recognition: {e}")
        st.session_state.user_speech = "" # Clear speech on error
    finally:
        status_placeholder.empty() # Clear the status message
        st.session_state.listening = False # Reset listening state
        # Rerun regardless of whether speech was captured or not,
        # to update the UI (e.g., button state) and process any captured speech.
        st.rerun()


# --- Input Area ---
input_area = st.container()
with input_area:
    # Determine input source (prioritize text input if both happen somehow)
    user_input_text = st.chat_input("Type your message here...", key="chat_input")
    user_input_from_speech = st.session_state.get("user_speech", "")

    user_input = user_input_text or user_input_from_speech
    input_method = 'speech' if user_input == user_input_from_speech and user_input != "" else 'text'

    # Process the input if there is any
    if user_input:
        # Clear speech input state immediately after capturing it for processing
        if input_method == 'speech':
            st.session_state.user_speech = ""

        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_input, "mode": st.session_state.current_mode})

        # Generate bot response
        with st.spinner("Thinking..."):
            bot_response = handle_user_input(user_input) # Process the user input

        # --- Text-to-Speech Generation ---
        # Generate TTS if enabled AND a response was generated
        should_speak = bot_response and st.session_state.get('tts_checkbox', True)

        if should_speak:
            # For diagnosis completion, use the full summary for TTS
            speak_text = bot_response # Default to the latest response
            if (st.session_state.current_mode == "diagnosis" and
                st.session_state.diagnosis_state.get("analysis_complete", False)):
                # Use the stored summary if available
                speak_text = st.session_state.diagnosis_state.get("summary", bot_response)

            # Generate audio and store it in session state to be played on the *next* rerun
            generate_and_store_tts(speak_text)
        else:
             # Ensure any previous audio is cleared if TTS is disabled or no response
             st.session_state.audio_to_play = None
             st.session_state.audio_key = None


        # Rerun AFTER processing and potentially generating TTS data.
        # This rerun will:
        # 1. Display the user's message.
        # 2. Display the assistant's message.
        # 3. Trigger the `st.audio` widget in the chat container if `audio_to_play` is set.
        st.rerun()
