# AI Programming Assistant Backend

A FastAPI backend that powers an AI programming assistant using Google's Gemini API.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd portfolio/backend
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key to `.env`

5. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API**
   - API docs: http://127.0.0.1:8000/docs
   - Health check: http://127.0.0.1:8000/health

## API Endpoints

- `POST /chat` - Send a message to the AI assistant
  - Request body: `{ "message": "your question here", "conversation_history": [] }`
  - Response: `{ "response": "AI response here", "is_code": boolean }`

- `GET /health` - Check if the API is running

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

## Development

- The server will automatically reload when you make changes to the code.
- Make sure to add new dependencies to `requirements.txt`.

## Testing

To test the API, you can use the interactive docs at `http://127.0.0.1:8000/docs` or use a tool like curl:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "How do I reverse a string in Python?",
    "conversation_history": []
  }'
```

## Deployment

For production deployment, consider using:
- Gunicorn with Uvicorn workers
- Environment variable management
- Proper CORS configuration
- HTTPS
- Rate limiting
- Monitoring and logging
