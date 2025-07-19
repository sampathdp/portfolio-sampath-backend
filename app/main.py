from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import traceback
import logging
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Programming Assistant API",
    description="API for AI-powered programming assistant using Gemini Free Tier",
    version="1.0.0"
)

# CORS middleware configuration
# Update this list with your frontend URL in production
allowed_origins = [
    "http://localhost:3000",  # Local development
    "http://127.0.0.1:3000",  # Local development alternative
    "https://portfolio-sampath-frontend.netlify.app/",  # Replace with your production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-1.5-flash for free tier (faster and more quota-friendly)
model = genai.GenerativeModel('gemini-1.5-flash')

# Enhanced rate limiter for free tier
class RateLimiter:
    def __init__(self, calls=15, period=60):  # 15 requests per minute for free tier
        self.calls = calls
        self.period = period
        self.timestamps = []

    def __call__(self):
        now = time.time()
        # Remove timestamps older than the period
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time + random.uniform(0.1, 0.5))  # Add jitter
        
        self.timestamps.append(time.time())

# Initialize rate limiter (15 requests per minute for free tier)
rate_limiter = RateLimiter(calls=15, period=60)

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    is_code: bool = False
    token_count: Optional[int] = None

# Enhanced system prompt for programming assistant
SYSTEM_PROMPT = """You are a helpful programming assistant for a campus project. 
Your responses should be:
- Clear and concise
- Include relevant code examples when appropriate
- Use proper code formatting with language-specific syntax highlighting
- Provide explanations for complex concepts
- Be educational and help students learn

When providing code:
- Use triple backticks with language specification (e.g., ```python)
- Include comments explaining key concepts
- Provide complete, runnable examples when possible
- Suggest best practices and explain why they're important

Keep responses focused and avoid unnecessary verbosity to optimize token usage.
"""

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    try:
        logger.info(f"Received chat request from IP: {request.client.host}")
        
        # Apply rate limiting first
        rate_limiter()
        
        # Build conversation context (limit to last 6 messages to save tokens)
        conversation_context = []
        
        # Add system prompt
        full_prompt = SYSTEM_PROMPT + "\n\nConversation history:\n"
        
        # Add recent conversation history
        for msg in chat_request.conversation_history[-6:]:
            full_prompt += f"{msg.role}: {msg.content}\n"
        
        # Add current message
        full_prompt += f"user: {chat_request.message}\n\nPlease respond as the programming assistant:"
        
        logger.info(f"Prompt length: {len(full_prompt)} characters")
        
        try:
            # Generate response with optimized settings for free tier
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,  # Reduced for free tier
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    stop_sequences=["user:", "assistant:"]
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
            )
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Check if response contains code blocks
            is_code = "```" in response_text
            
            # Try to get token count (if available)
            token_count = None
            if hasattr(response, 'usage_metadata'):
                token_count = response.usage_metadata.total_token_count
            
            logger.info(f"Generated response: {len(response_text)} characters, tokens: {token_count}")
            
            return ChatResponse(
                response=response_text,
                is_code=is_code,
                token_count=token_count
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error generating content: {e}")
            
            # Handle different types of errors
            if "quota" in error_msg or "rate limit" in error_msg:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "API quota exceeded. Please wait a moment before sending another message.",
                        "error": "Rate limit exceeded",
                        "retry_after": 60
                    }
                )
            elif "safety" in error_msg:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Content was blocked by safety filters. Please rephrase your question.",
                        "error": "Content filtered"
                    }
                )
            elif "token" in error_msg:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Message is too long. Please shorten your request.",
                        "error": "Token limit exceeded"
                    }
                )
            else:
                raise
        
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while processing your request",
                "error": str(e)
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "gemini-1.5-flash",
        "tier": "free"
    }

# Usage stats endpoint (helpful for monitoring free tier usage)
@app.get("/stats")
async def get_stats():
    return {
        "rate_limiter_calls": len(rate_limiter.timestamps),
        "requests_in_last_minute": len([t for t in rate_limiter.timestamps if time.time() - t < 60]),
        "uptime": "healthy"
    }

# Clear conversation endpoint
@app.post("/clear")
async def clear_conversation():
    return {"message": "Conversation cleared (client-side implementation needed)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)