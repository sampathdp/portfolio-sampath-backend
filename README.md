# AI Programming Assistant Backend

A FastAPI backend that powers an AI programming assistant using Google's Gemini API.

## Local Development Setup

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

## Deployment to Railway

### Prerequisites
- A Railway account (https://railway.app/)
- A GitHub account with your repository connected
- Your Google Gemini API key

### Deployment Steps

1. **Push your code to GitHub**
   Make sure all your changes are committed and pushed to your GitHub repository.

2. **Create a new Railway project**
   - Go to https://railway.app/dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Select your repository and the branch you want to deploy
   - Click "Deploy Now"

3. **Set up environment variables**
   - In your Railway project dashboard, go to the "Variables" tab
   - Add your `GEMINI_API_KEY` with your actual Gemini API key
   - Add `PORT` variable with value `8000` (Railway will automatically assign a port, but we include this as a fallback)

4. **Configure deployment settings**
   - Railway will automatically detect this is a Python project and use the `Procfile`
   - The build process will install all dependencies from `requirements.txt`
   - The server will start using the command in the `Procfile`

5. **Access your deployed API**
   - Once deployed, Railway will provide you with a public URL
   - Access the API docs at `https://your-railway-url.railway.app/docs`
   - Health check endpoint: `https://your-railway-url.railway.app/health`

### Custom Domain (Optional)
If you want to use a custom domain:
1. Go to the "Settings" tab in your Railway project
2. Click on "Generate Domain" under "Domains"
3. Follow the instructions to set up a custom domain

### Monitoring and Logs
- Check the "Logs" tab in Railway for real-time logs
- Monitor resource usage in the "Metrics" tab

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
