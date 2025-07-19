# Standard library imports
import logging
import os
import time
from typing import List, Optional

# Third-party imports
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
# Allow all origins for now to test the connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
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

class RateLimiter:
    """
    Rate limiter to control the number of API calls within a specific time period.
    Defaults to 15 requests per minute (Gemini free tier limit).
    """
    def __init__(self, calls: int = 15, period: int = 60) -> None:
        """
        Initialize the rate limiter.
        
        Args:
            calls: Maximum number of allowed calls within the period
            period: Time period in seconds for rate limiting
        """
        self.calls = max(1, calls)  # Ensure at least 1 call is allowed
        self.period = max(1, period)  # Ensure period is at least 1 second
        self.timestamps: List[float] = []
        self.lock = False  # Simple lock to prevent race conditions

    def __call__(self) -> None:
        """
        Check if the current call is allowed based on rate limits.
        
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Simple lock to prevent race conditions in a multi-threaded environment
        while self.lock:
            time.sleep(0.01)  # Small delay to prevent CPU spinning
            
        self.lock = True
        try:
            now = time.time()
            
            # Remove timestamps older than the period
            self.timestamps = [t for t in self.timestamps if now - t < self.period]
            
            if len(self.timestamps) >= self.calls:
                retry_after = int(self.period - (now - self.timestamps[0]))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={"Retry-After": str(retry_after)},
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": retry_after,
                        "limit": self.calls,
                        "period_seconds": self.period
                    }
                )
            
            self.timestamps.append(now)
            
        finally:
            self.lock = False

# Initialize rate limiter (15 requests per minute for free tier)
rate_limiter = RateLimiter(calls=15, period=60)

# Request/Response models
class ChatMessage(BaseModel):
    """Represents a single message in the chat conversation."""
    role: str = Field(..., description="The role of the message sender (user/assistant)")
    content: str = Field(..., min_length=1, description="The content of the message")

    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "content": "How do I write a for loop in Python?"
            }
        }

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="The user's message")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="List of previous messages in the conversation"
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "How do I write a for loop in Python?",
                "conversation_history": []
            }
        }
    }

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="The assistant's response")
    is_code: bool = Field(
        default=False,
        description="Indicates if the response contains code that should be formatted"
    )
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens used in the response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Here's how to write a for loop in Python:\n```python\nfor i in range(5):\n    print(i)\n```",
                "is_code": True,
                "token_count": 42
            }
        }

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