# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import os
import json
import uuid
from pydantic import ConfigDict
import datetime
from datetime import date
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
# Near the top of your app.py file, add this:
import os
import re
from pydantic import ValidationError
from dotenv import load_dotenv
from groq import Groq  # Changed to use Groq client directly


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Healthcare Chatbot API")

# Check if the API key is present
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("WARNING: GROQ_API_KEY not found in environment variables. LLM calls will fail.")

# Function to get LLM instance with specific parameters
def get_llm(temperature: float = 0.1, max_tokens: Optional[int] = 500, model: str = "mixtral-8x7b-32768"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set")
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Please set this environment variable."
        )
    
    print(f"Using Groq API key: {groq_api_key[:5]}..." if groq_api_key else "No API key found")
    
    try:
        return Groq(api_key=groq_api_key)
    except Exception as e:
        print(f"ERROR initializing Groq client: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM: {str(e)}"
        )

def get_groq_client():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set")
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Please set this environment variable."
        )
    
    print(f"Using Groq API key: {groq_api_key[:5]}..." if groq_api_key else "No API key found")
    
    try:
        return Groq(api_key=groq_api_key)
    except Exception as e:
        print(f"ERROR initializing Groq client: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM: {str(e)}"
        )


# Load hospital layout information
with open("hospital_layout.txt", "r") as f:
    HOSPITAL_LAYOUT = f.read()

# Define the system prompt templates
HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare chatbot designed to provide helpful and comforting information to patients. 
You must respond in the following JSON format only:
{
    "chatbot_answer": "<your detailed and empathetic healthcare response>",
    "category": "<one of: require_human_intervention, require_medicine, personal_assistance_needed, hospital_info>"
}

Guidelines for responses:
- chatbot_answer: Provide clear, compassionate, and helpful healthcare information
- category must be exactly one of these four options:
  * require_human_intervention: For serious medical concerns requiring immediate professional attention
  * require_medicine: For conditions that might need medication or pharmacy advice
  * personal_assistance_needed: For situations requiring caregiver or personal support
  * hospital_info: For general hospital/clinic information queries

The response must be valid JSON. Do not include any text outside the JSON structure."""

CONVERSATION_SYSTEM_PROMPT = """You are a friendly and empathetic healthcare chatbot designed to provide comfort and 
companionship to patients who might be feeling low or anxious. Engage in a warm, supportive conversation with the patient.
Your responses should be brief (1-3 sentences), empathetic, and encouraging without giving specific medical advice.
Focus on being a good listener and asking thoughtful follow-up questions to show you care."""

NAVIGATION_SYSTEM_PROMPT = f"""You are a hospital navigation assistant. Your goal is to help patients and visitors
find their way around the hospital based on the hospital layout information provided below. Your responses should be clear,
concise, and helpful. When giving directions, provide step-by-step instructions using landmarks and room numbers.

Hospital Layout Information:
{HOSPITAL_LAYOUT}

Always respond in a friendly tone and ask if they need further clarification on the directions."""

# Update the DIAGNOSIS_SYSTEM_PROMPT in app.py

DIAGNOSIS_SYSTEM_PROMPT = """You are a medical diagnosis assistant. You MUST respond using ONLY this exact JSON format:
{
    "response": "Your question or response text here",
    "current_question": number,
    "status": "in_progress|complete",
    "patient_info": {
        "age": "",
        "gender": "",
        "weight": "",
        "height": "",
        "blood_group": ""
    },
    "symptoms": {
        "main_symptoms": [],
        "duration": "",
        "severity": ""
    },
    "medical_history": {
        "conditions": [],
        "medications": []
    }
}

Follow these strict guidelines:
1. ALWAYS maintain the EXACT JSON structure shown above
2. Only update fields when you get new information
3. The 'response' field should contain your next question or acknowledgment
4. Increment current_question after each patient response
5. Follow this sequence of questions:
   - Question 1: Ask for age
   - Question 2: Ask for gender
   - Question 3: Ask for weight and height
   - Question 4: Ask for main symptoms
   - Question 5: Ask for symptom duration
   - Question 6: Ask for symptom severity
   - Question 7: Ask for existing medical conditions
   - Question 8: Ask for current medications
   - Question 9: Ask for any other relevant information
   - Question 10: Provide a summary of collected information
6. Set status to 'complete' ONLY after question 10
7. NEVER add fields that are not in the template
8. NEVER send a response that is not valid JSON"""



class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (system, user)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Controls randomness in the response"
    )
    model: str = Field(
        default="mixtral-8x7b-32768",
        description="The language model to use"
    )
    mode: str = Field(
        default="healthcare",
        description="Mode of conversation: healthcare, conversation, navigation, or diagnosis"
    )
    diagnosis_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="State of the diagnosis conversation"
    )

class BaseResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthcareResponse(BaseResponse):
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "category": "hospital_info",
        "requires_attention": False
    })

class ConversationResponse(BaseResponse):
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "sentiment": "neutral",
        "follow_up": True
    })

class NavigationResponse(BaseResponse):
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "location": "",
        "estimated_time": ""
    })

class DiagnosisResponse(BaseResponse):
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "current_question": 1,  # Changed from string "1" to integer 1
        "call_doctor": False,
        "patient_condition": "pending",
        "patient_info": {},
        "symptoms": {},
        "medical_history": {}
    })

class ChatResponse(BaseModel):
    response: Any
    model_name: str
    


# Store for diagnosis sessions
diagnosis_sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    try:
        print(f"Received request in mode: {request.mode}")
        
        # Get Groq client
        client = get_groq_client()
        
        # Set up system prompt based on mode
        if request.mode == "healthcare":
            system_prompt = HEALTHCARE_SYSTEM_PROMPT
        elif request.mode == "conversation":
            system_prompt = CONVERSATION_SYSTEM_PROMPT
        elif request.mode == "navigation":
            system_prompt = NAVIGATION_SYSTEM_PROMPT
        elif request.mode == "diagnosis":
            system_prompt = DIAGNOSIS_SYSTEM_PROMPT
        else:
            raise HTTPException(status_code=400, detail="Invalid mode specified")
        
        print(f"Using system prompt for mode: {request.mode}")
        
        # Format messages for Groq API
        groq_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add user message
        for msg in request.messages:
            if msg.role == "user":
                groq_messages.append({"role": msg.role, "content": msg.content})
                print(f"User message: {msg.content}")
        
        # Get response from Groq
        print("Calling Groq API...")
        chat_completion = client.chat.completions.create(
            messages=groq_messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=500,  # You can adjust this as needed
        )
        
        # Extract content from response
        ai_content = chat_completion.choices[0].message.content
        print(f"Groq response received: {ai_content[:100]}...")
        
        # Process the response based on mode
        if request.mode == "healthcare":
            try:
                response_dict = json.loads(ai_content)
                response = HealthcareResponse(
                    response=response_dict["chatbot_answer"],
                    metadata={
                        "category": response_dict["category"],
                        "requires_attention": response_dict["category"] == "require_human_intervention"
                    }
                )
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Failed to parse LLM response")

        elif request.mode == "conversation":
            response = ConversationResponse(
                response=ai_content,
                metadata={
                    "sentiment": "friendly",  # You could add sentiment analysis here
                    "follow_up": True
                }
            )

        elif request.mode == "navigation":
            response = NavigationResponse(
                response=ai_content,
                metadata={
                    "location": "Main Building",  # You could extract location info
                    "estimated_time": "2 minutes"
                }
            )

        # In the create_chat_completion function, update the diagnosis section:
        elif request.mode == "diagnosis":
            try:
                # Try to parse the AI response as JSON
                cleaned_response = ai_content.strip()
                
                # Handle JSON code blocks if present
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
                
                # Default response structure to use if parsing fails
                default_response = {
                    "response": "I'm having trouble processing your information. Could you please tell me your age?",
                    "current_question": 1,
                    "status": "in_progress",
                    "patient_info": {},
                    "symptoms": {},
                    "medical_history": {}
                }
                
                # Try to parse the cleaned JSON with extensive error handling
                try:
                    response_dict = json.loads(cleaned_response)
                    print(f"Successfully parsed diagnosis JSON response")
                    
                    # Safety check for current_question field
                    if "current_question" not in response_dict or response_dict["current_question"] is None:
                        response_dict["current_question"] = 1
                    
                    # Type conversion for current_question with extensive error handling
                    if isinstance(response_dict["current_question"], dict):
                        print("WARNING: current_question is a dict, setting to default value 1")
                        response_dict["current_question"] = 1
                    elif isinstance(response_dict["current_question"], str):
                        try:
                            response_dict["current_question"] = int(response_dict["current_question"])
                        except ValueError:
                            response_dict["current_question"] = 1
                    
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"Error processing diagnosis response: {str(e)}")
                    print(f"Failed JSON text: {cleaned_response}")
                    response_dict = default_response
                
                # Create DiagnosisResponse with very cautious type checking
                try:
                    current_q = response_dict.get("current_question", 1)
                    # Force current_question to be an integer
                    if not isinstance(current_q, int):
                        if isinstance(current_q, str) and current_q.isdigit():
                            current_q = int(current_q)
                        else:
                            current_q = 1
                    
                    # Ensure we have valid dictionaries for nested fields
                    patient_info = response_dict.get("patient_info", {})
                    if not isinstance(patient_info, dict):
                        patient_info = {}
                        
                    symptoms = response_dict.get("symptoms", {})
                    if not isinstance(symptoms, dict):
                        symptoms = {}
                        
                    medical_history = response_dict.get("medical_history", {})
                    if not isinstance(medical_history, dict):
                        medical_history = {}
                    
                    response = DiagnosisResponse(
                        response=response_dict.get("response", "Could you tell me your age?"),
                        metadata={
                            "current_question": current_q,
                            "status": response_dict.get("status", "in_progress"),
                            "patient_info": patient_info,
                            "symptoms": symptoms,
                            "medical_history": medical_history,
                            "timestamp": str(datetime.datetime.now())  # Convert datetime to string to avoid serialization issues
                        }
                    )
                except Exception as e:
                    print(f"Error creating DiagnosisResponse: {str(e)}")
                    # Extremely simplified fallback response
                    response = DiagnosisResponse(
                        response="I encountered a technical issue with our diagnosis system. Could we start again with your age?",
                        metadata={
                            "current_question": 1, 
                            "status": "in_progress",
                            "patient_info": {},
                            "symptoms": {},
                            "medical_history": {}
                        }
                    )
            
            except Exception as e:
                print(f"Critical Diagnosis Error: {str(e)}")
                print(f"Raw AI Response: {ai_content[:200]}...")
                # Return a more informative default response on error
                response = DiagnosisResponse(
                    response="I encountered a technical issue with our diagnosis system. Could we start again with your age?",
                    metadata={
                        "current_question": 1, 
                        "status": "in_progress",
                        "patient_info": {},
                        "symptoms": {},
                        "medical_history": {}
                    }
                )

        return ChatResponse(
            response=response,
            model_name=request.model
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Healthcheck endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)