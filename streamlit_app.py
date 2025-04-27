# streamlit_app.py
import streamlit as st
import requests
import json
import uuid
import base64
from io import BytesIO
import datetime
import os
import speech_recognition as sr
from gtts import gTTS

# Set up page configuration
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def validate_diagnosis_input(current_q, user_input):
    """Validate user input for diagnosis questions"""
    if not user_input or user_input.isspace():
        return "Please provide an answer"

    if current_q == 1:  # Age
        if not user_input.isdigit() or not 0 < int(user_input) < 150:
            return "Please enter a valid age between 1 and 150"
    elif current_q == 2:  # Gender
        if user_input.lower() not in ['male', 'female', 'other']:
            return "Please enter Male, Female, or Other"
    elif current_q == 3:  # Weight
        try:
            weight = float(user_input)
            if not 1 <= weight <= 500:
                return "Please enter a valid weight between 1 and 500 kg"
        except ValueError:
            return "Please enter a valid number for weight"
    elif current_q == 8:  # Severity
        if not user_input.isdigit() or not 1 <= int(user_input) <= 10:
            return "Please enter a number between 1 and 10"
    
    return None


def generate_diagnosis_report(diagnosis_state):
    """Generate and save diagnosis report"""
    # Helper function to safely convert lists to string
    def safe_join(items, separator=', '):
        if not items:
            return ""
        # Convert any non-string items to strings
        return separator.join(str(item) for item in items)
    
    report = f"""
MEDICAL CONSULTATION REPORT
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {diagnosis_state['session_id']}

PATIENT INFORMATION
-----------------
Age: {diagnosis_state['patient_info'].get('age', 'Not provided')}
Gender: {diagnosis_state['patient_info'].get('gender', 'Not provided')}
Weight: {diagnosis_state['patient_info'].get('weight', 'Not provided')} kg
Height: {diagnosis_state['patient_info'].get('height', 'Not provided')} cm
Blood Group: {diagnosis_state['patient_info'].get('blood_group', 'Not provided')}

SYMPTOMS AND CONDITION
-------------------
Main Symptoms: {safe_join(diagnosis_state['symptoms'].get('main_symptoms', []))}
Duration: {diagnosis_state['symptoms'].get('duration', 'Not specified')}
Severity (1-10): {diagnosis_state['symptoms'].get('severity', 'Not specified')}

MEDICAL HISTORY
-------------
Existing Conditions: {safe_join(diagnosis_state['medical_history'].get('conditions', []))}
Current Medications: {safe_join(diagnosis_state['medical_history'].get('medications', []))}

ASSESSMENT
---------
Condition: {diagnosis_state.get('patient_condition', 'PENDING').upper()}
Requires Immediate Medical Attention: {'Yes' if diagnosis_state.get('call_doctor', False) else 'No'}

RECOMMENDATIONS
-------------
{generate_recommendations(diagnosis_state)}
"""
    
    # Create output directory if it doesn't exist
    os.makedirs('diagnosis_reports', exist_ok=True)
    
    # Save report
    filename = f"diagnosis_reports/report_{diagnosis_state['session_id']}.txt"
    with open(filename, 'w') as f:
        f.write(report)
    
    return report, filename

def generate_recommendations(diagnosis_state):
    """Generate recommendations based on diagnosis state"""
    if diagnosis_state['call_doctor']:
        return "URGENT: Please seek immediate medical attention."
    
    severity = int(diagnosis_state['symptoms'].get('severity', 0))
    recommendations = []
    
    if severity >= 7:
        recommendations.append("- Schedule an appointment with a doctor soon")
    elif severity >= 4:
        recommendations.append("- Monitor symptoms closely")
        recommendations.append("- If symptoms worsen, consult a healthcare provider")
    else:
        recommendations.append("- Rest and monitor symptoms")
        recommendations.append("- Practice self-care measures")
    
    return "\n".join(recommendations)

# ---------- Session State Initialization ----------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "conversation"

if 'diagnosis_state' not in st.session_state:
    st.session_state.diagnosis_state = {
        "session_id": str(uuid.uuid4()),
        "current_question": 1,  # Track question number as integer
        "patient_info": {},
        "symptoms": {},
        "medical_history": {},
        "call_doctor": False,
        "patient_condition": "pending",
        "last_question": "",
        "show_results": False  # Add this flag
    }

if 'speaking' not in st.session_state:
    st.session_state.speaking = False

if 'listening' not in st.session_state:
    st.session_state.listening = False

if 'user_speech' not in st.session_state:
    st.session_state.user_speech = ""

# ---------- Constants & Configuration ----------
API_URL = "http://localhost:8000/chat"
# Static questions with validation rules
DIAGNOSIS_QUESTIONS = {
    1: {"text": "Please tell me your age:", "field": "age", "category": "patient_info"},
    2: {"text": "What is your gender? (Male/Female/Other):", "field": "gender", "category": "patient_info"},
    3: {"text": "What is your weight in kg?", "field": "weight", "category": "patient_info"},
    4: {"text": "What is your height in cm?", "field": "height", "category": "patient_info"},
    5: {"text": "What is your blood group?", "field": "blood_group", "category": "patient_info"},
    6: {"text": "What symptoms are you experiencing?", "field": "main_symptoms", "category": "symptoms"},
    7: {"text": "How long have you had these symptoms?", "field": "duration", "category": "symptoms"},
    8: {"text": "On a scale of 1-10, how severe is your discomfort?", "field": "severity", "category": "symptoms"},
    9: {"text": "Do you have any existing medical conditions?", "field": "conditions", "category": "medical_history"},
    10: {"text": "Are you currently taking any medications?", "field": "medications", "category": "medical_history"}
}

# ---------- UI Configuration ----------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stChatMessage { background-color: #1e2530; border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stTextInput>div>div>input { border-radius: 25px; background-color: #262730; color: white; }
    .stButton>button { border-radius: 50%; height: 50px; width: 50px; font-size: 24px; }
    .diagnosis-progress { padding: 1rem; border-radius: 10px; background: #1e2530; }
</style>
""", unsafe_allow_html=True)

# ---------- Core Functions ----------
def validate_diagnosis_answer(question_num, answer):
    """Validate user input based on question type"""
    answer = str(answer).strip()
    
    if not answer:
        return "Please provide an answer"
        
    if question_num == 1:  # Age
        if not answer.isdigit() or not 0 < int(answer) < 150:
            return "Please enter a valid age (0-150)"
    elif question_num == 2:  # Gender
        if answer.lower() not in ['male', 'female', 'other']:
            return "Please enter Male, Female, or Other"
    elif question_num == 3:  # Weight
        try:
            weight = float(answer)
            if not 1 <= weight <= 500:
                return "Please enter a valid weight in kg (1-500)"
        except ValueError:
            return "Please enter a valid number for weight"
    elif question_num == 4:  # Height
        try:
            height = float(answer)
            if not 30 <= height <= 300:
                return "Please enter a valid height in cm (30-300)"
        except ValueError:
            return "Please enter a valid number for height"
    elif question_num == 8:  # Severity
        if not answer.isdigit() or not 1 <= int(answer) <= 10:
            return "Please enter a number between 1 and 10"
            
    return None

def analyze_diagnosis_results():
    """Analyze collected diagnosis data and determine if doctor is needed"""
    try:
        # First, log the current state for debugging
        print("Analyzing diagnosis results...")
        print(f"Current diagnosis state: {st.session_state.diagnosis_state}")
        
        # Extract severity with careful type handling
        severity_val = st.session_state.diagnosis_state.get("symptoms", {}).get("severity", "0")
        try:
            # Try to convert to int, defaulting to 0 if it fails
            severity = int(severity_val) if isinstance(severity_val, (str, int)) else 0
        except (ValueError, TypeError):
            severity = 0
            
        print(f"Severity determined to be: {severity}")
        
        # Make assessment based on severity
        call_doctor = severity >= 8
        if severity >= 8:
            condition = "severe"
        elif severity >= 5:
            condition = "moderate" 
        else:
            condition = "mild"
            
        # Update diagnosis state with assessment
        st.session_state.diagnosis_state.update({
            "call_doctor": call_doctor,
            "patient_condition": condition,
            "analysis_complete": True  # This flag is important!
        })
        
        print(f"Analysis complete. Condition: {condition}, Call doctor: {call_doctor}")
        return f"Analysis complete. Your condition appears to be {condition}."
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        st.error(f"Analysis error: {str(e)}")
        return f"Error analyzing results: {str(e)}"
    
def send_for_analysis(diagnosis_state):
    """Send collected data to API for final analysis"""
    try:
        # Format data for analysis
        analysis_request = {
            "patient_data": {
                "age": int(diagnosis_state["patient_info"].get("age", 0)),
                "gender": diagnosis_state["patient_info"].get("gender", ""),
                "weight": float(diagnosis_state["patient_info"].get("weight", 0)),
                "height": float(diagnosis_state["patient_info"].get("height", 0)),
                "blood_group": diagnosis_state["patient_info"].get("blood_group", "")
            },
            "symptoms": {
                "main_symptoms": diagnosis_state["symptoms"].get("main_symptoms", []),
                "duration": diagnosis_state["symptoms"].get("duration", ""),
                "severity": int(diagnosis_state["symptoms"].get("severity", 0))
            },
            "medical_history": {
                "conditions": diagnosis_state["medical_history"].get("conditions", []),
                "medications": diagnosis_state["medical_history"].get("medications", [])
            }
        }

        # Send to API for analysis
        response = requests.post(
            API_URL,
            json={
                "messages": [{
                    "role": "system",
                    "content": "Analyze the following patient data and provide recommendations."
                }, {
                    "role": "user",
                    "content": json.dumps(analysis_request)
                }],
                "temperature": 0.2,
                "model": "mixtral-8x7b-32768",
                "mode": "analysis"
            }
        ).json()

        # Update diagnosis state with analysis results
        if "response" in response:
            try:
                analysis = json.loads(response["response"])
                diagnosis_state.update({
                    "call_doctor": analysis.get("call_doctor", False),
                    "patient_condition": analysis.get("patient_condition", "pending"),
                    "analysis": analysis.get("analysis", ""),
                    "recommendations": analysis.get("recommendations", [])
                })
            except json.JSONDecodeError:
                # Fallback to severity-based analysis
                severity = int(diagnosis_state["symptoms"].get("severity", 0))
                diagnosis_state.update({
                    "call_doctor": severity >= 8,
                    "patient_condition": "severe" if severity >= 8 else "moderate" if severity >= 5 else "mild",
                    "analysis": f"Based on severity level {severity}",
                    "recommendations": ["Seek medical attention" if severity >= 8 else "Monitor symptoms"]
                })

        return diagnosis_state

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return diagnosis_state

def handle_diagnosis_response(user_input):
    """Handle diagnosis flow with static question progression"""
    try:
        # Get current question
        current_q = st.session_state.diagnosis_state["current_question"]
        
        # Store the user's answer
        question_info = DIAGNOSIS_QUESTIONS[current_q]
        category = question_info["category"]
        field = question_info["field"]
        
        # Ensure category exists in state
        if category not in st.session_state.diagnosis_state:
            st.session_state.diagnosis_state[category] = {}
        
        # Store the answer with proper handling for array fields
        if field in ["main_symptoms", "conditions", "medications"]:
            # For list fields, convert to string and store as single item list
            user_input_str = str(user_input)
            st.session_state.diagnosis_state[category][field] = [user_input_str]
        else:
            # For scalar fields, store as string to avoid type issues
            st.session_state.diagnosis_state[category][field] = str(user_input)

        # Move to next question
        next_q = current_q + 1
        
        # If all questions are answered, send data for analysis
        if next_q > 10:
            # Validate that we have all necessary categories
            for cat in ["patient_info", "symptoms", "medical_history"]:
                if cat not in st.session_state.diagnosis_state:
                    st.session_state.diagnosis_state[cat] = {}
            
            # No need to convert anything else here - analysis will handle conversions
            analysis_result = analyze_diagnosis_results()
            return "Analysis complete. Here are your results."
        
        # Update current question and return next question
        st.session_state.diagnosis_state["current_question"] = next_q
        return DIAGNOSIS_QUESTIONS[next_q]["text"]

    except Exception as e:
        st.error(f"Diagnosis response error: {str(e)}")
        # Include the specific error for debugging
        return f"Error processing your answer: {str(e)}. Let's try again with the current question."

def process_response(response_data, mode):
    """Handle API responses for all modes"""
    if not response_data or "response" not in response_data:
        return "No response from server. Please try again."
    
    response = response_data.get("response", {})
    
    # Extract the actual response text
    if isinstance(response, dict):
        if mode == "healthcare":
            text = response.get("response", "")  # Get response from healthcare format
        elif mode == "diagnosis":
            return handle_diagnosis_response(response_data)
        else:
            text = response.get("response", "")  # Default response field
    else:
        text = str(response)  # Convert response to string if it's not a dict
    
    # Ensure we have a valid response
    if not text or text.isspace():
        if mode == "healthcare":
            return "I apologize, but I couldn't process that request. Could you please rephrase?"
        elif mode == "conversation":
            return "I'm here to listen. Could you tell me more?"
        elif mode == "navigation":
            return "Could you please specify where you'd like to go in the hospital?"
        else:
            return "I didn't understand that. Could you please try again?"
    
    return text

# ---------- UI Components ----------
def show_diagnosis_progress():
    """Display diagnosis progress and current question"""
    try:
        diagnosis_state = st.session_state.diagnosis_state
        current_q = diagnosis_state["current_question"]
        
        with st.container(border=True):
            st.markdown("### Health Assessment Progress")
            
            # Progress bar
            if 1 <= current_q <= 10:
                progress = current_q / 10
                st.progress(progress, text=f"Question {current_q}/10")
                st.caption(DIAGNOSIS_QUESTIONS[current_q]["text"])
            
            # Show results when completed
            if current_q > 10 or diagnosis_state.get("show_results", False):
                # Print debug information
                print("Showing diagnosis results")
                print(f"Current state: {diagnosis_state}")
                show_diagnosis_results()

    except Exception as e:
        st.error(f"Progress display error: {str(e)}")
        print(f"Error in progress display: {str(e)}")
        
def get_conversation_context():
    """Get the current conversation context including previous messages"""
    messages = []
    for msg in st.session_state.messages[-5:]:  # Get last 5 messages for context
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return messages
def show_diagnosis_results():
    """Display final assessment results and generate report"""
    ds = st.session_state.diagnosis_state
    
    # Generate and save report
    report, filename = generate_diagnosis_report(ds)
    
    st.divider()
    st.subheader("Assessment Result")
    
    if ds["call_doctor"]:
        st.error("⚠️ Immediate Medical Attention Recommended")
        st.write("A healthcare professional will contact you shortly.")
    else:
        st.success("✅ Assessment Complete")
    
    # Display report in expandable section
    with st.expander("View Detailed Report", expanded=True):
        st.text(report)
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"medical_report_{ds['session_id']}.txt",
            mime="text/plain"
        )
    
    if st.button("Start New Assessment"):
        st.session_state.diagnosis_state = {
            "session_id": str(uuid.uuid4()),
            "current_question": 1,
            "patient_info": {},
            "symptoms": {},
            "medical_history": {},
            "call_doctor": False,
            "patient_condition": "pending",
            "last_question": ""
        }
        st.rerun()

# ---------- Main App Layout ----------
st.title("🏥 Hospital Healthcare Assistant")

# Sidebar
with st.sidebar:
    st.header("Assistant Mode")
    mode = st.radio(
        "Select mode",
        ["Friendly Conversation", "Hospital Navigation", "Health Assessment", "Medical Inquiry"],
        index=0
    )
    st.session_state.current_mode = {
        "Friendly Conversation": "conversation",
        "Hospital Navigation": "navigation",
        "Health Assessment": "diagnosis",
        "Medical Inquiry": "healthcare"
    }[mode]
    
    st.divider()
    st.subheader("Audio Controls")
    text_to_speech = st.checkbox("Enable Text-to-Speech", value=True)
    speech_to_text = st.checkbox("Enable Speech-to-Text", value=True)
    
    st.divider()
    st.subheader("Model Selection")
    model = st.selectbox(
        "Select model from Groq Cloud",
        ["llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )

if st.session_state.current_mode == "diagnosis":
    show_diagnosis_progress()

# Chat Interface
st.subheader("Conversation")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])



# Chat Interface
st.subheader("Conversation")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ---------- Input Handling ----------
def handle_user_input(user_input):
    """Process user input and generate responses"""
    # Special handling for diagnosis mode
    if st.session_state.current_mode == "diagnosis":
        current_q = st.session_state.diagnosis_state["current_question"]
        
        # Add to conversation history for display
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Validate input first
        validation_msg = validate_diagnosis_answer(current_q, user_input)
        if validation_msg:
            response = f"{validation_msg}\n\n{DIAGNOSIS_QUESTIONS[current_q]['text']}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            return response
            
        # Process the response locally without API
        try:
            # Get question info
            question_info = DIAGNOSIS_QUESTIONS[current_q]
            category = question_info["category"]
            field = question_info["field"]
            
            # Ensure category exists in state
            if category not in st.session_state.diagnosis_state:
                st.session_state.diagnosis_state[category] = {}
            
            # Store answer with proper type handling
            if field in ["main_symptoms", "conditions", "medications"]:
                # For list fields, store as array
                st.session_state.diagnosis_state[category][field] = [user_input]
            else:
                # For scalar fields, store directly
                st.session_state.diagnosis_state[category][field] = user_input
                
            # Check if we're done with questions
            next_q = current_q + 1
            if next_q > len(DIAGNOSIS_QUESTIONS):
                # Complete the diagnosis flow
                st.session_state.diagnosis_state["current_question"] = next_q
                
                # Run analysis locally
                analysis_result = analyze_diagnosis_results()
                
                # Generate a detailed assessment message for the chat
                condition = st.session_state.diagnosis_state.get("patient_condition", "undefined").upper()
                call_doctor = st.session_state.diagnosis_state.get("call_doctor", False)
                
                # Create a summary message for the chat
                summary = f"""
Thank you for providing all the information. Here's my assessment:

ASSESSMENT:
---------
Your condition appears to be: {condition}
Need immediate medical attention: {"Yes" if call_doctor else "No"}

RECOMMENDATIONS:
-------------
{generate_recommendations(st.session_state.diagnosis_state)}

A detailed report has been generated for your records.
                """
                
                # Add the detailed response to chat
                st.session_state.messages.append({"role": "assistant", "content": summary})
                
                # Set flag to show detailed results in the UI
                st.session_state.diagnosis_state["show_results"] = True
                st.session_state.diagnosis_state["summary"] = summary  # Store for TTS
                
                # Force a rerun to show results
                st.rerun()
                return summary  # Return summary for TTS
            
            # Move to next question
            st.session_state.diagnosis_state["current_question"] = next_q
            response = DIAGNOSIS_QUESTIONS[next_q]["text"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            return response
            
        except Exception as e:
            response = f"Error in diagnosis: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            return response
    
    # Rest of the function for other modes...
    try:
        response_data = requests.post(
            API_URL,
            json={
                "messages": get_conversation_context() + [{"role": "user", "content": user_input}],
                "temperature": 0.2,
                "model": model,
                "mode": st.session_state.current_mode
            }
        ).json()
        
        response = process_response(response_data, st.session_state.current_mode)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return response
        
    except Exception as e:
        response = f"API error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        return response

# Voice Input
if speech_to_text and not st.session_state.listening:
    if st.button("🎤", help="Click to speak"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.session_state.listening = True
                audio = r.listen(source)
                text = r.recognize_google(audio)
                st.session_state.user_speech = text
        except Exception as e:
            st.session_state.user_speech = f"Error: {str(e)}"
        finally:
            st.session_state.listening = False

# Text Input
# Text Input and Response handling 
user_input = st.chat_input("Type your message here...") or st.session_state.user_speech
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
        
    # Generate bot response (will add to messages internally)
    bot_response = handle_user_input(user_input)
    
    # Display assistant message (only need this if we didn't already add to messages)
    with st.chat_message("assistant"):
        st.write(bot_response)
    
    # Show diagnosis results if needed
    if st.session_state.current_mode == "diagnosis":
        current_q = st.session_state.diagnosis_state["current_question"]
        if current_q > len(DIAGNOSIS_QUESTIONS) and st.session_state.diagnosis_state.get("analysis_complete", False):
            # Results are displayed in chat, and the detailed view is shown
            show_diagnosis_results()
    
    # Text-to-Speech
    if text_to_speech and bot_response:
        # For diagnosis completion, use the full assessment for TTS
        if (st.session_state.current_mode == "diagnosis" and 
            st.session_state.diagnosis_state.get("current_question", 0) > len(DIAGNOSIS_QUESTIONS) and
            st.session_state.diagnosis_state.get("analysis_complete", False)):
            # Use the summary we stored for TTS
            speak_text = st.session_state.diagnosis_state.get("summary", bot_response)
        else:
            # Otherwise use the normal response
            speak_text = bot_response
            
        # Generate and play audio
        audio_file = gTTS(text=speak_text, lang='en').write_to_fp(fp := BytesIO())
        st.markdown(f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{base64.b64encode(fp.getvalue()).decode()}">
            </audio>
        """, unsafe_allow_html=True)
    
    # Clear voice input
    if user_input == st.session_state.user_speech:
        st.session_state.user_speech = ""