import requests
import json

# Update with your FastAPI server URL
BASE_URL = "http://127.0.0.1:8000"

def test_chat():
    """Test the chat endpoint with a sample programming question."""
    url = f"{BASE_URL}/chat"
    headers = {"Content-Type": "application/json"}
    
    # Sample programming questions to test
    questions = [
        "How do I reverse a string in Python?",
        "Explain async/await in JavaScript with an example.",
        "What's the difference between a list and a tuple in Python?"
    ]
    
    conversation_history = []
    
    for question in questions:
        print(f"\nYou: {question}")
        
        # Prepare the request payload
        payload = {
            "message": question,
            "conversation_history": conversation_history
        }
        
        try:
            # Send the request
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # Get the response data
            data = response.json()
            
            # Print the response
            print("\nAssistant:")
            print(data['response'])
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": data['response']})
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            break

if __name__ == "__main__":
    print("Testing AI Programming Assistant API...")
    print("Make sure the FastAPI server is running before testing!")
    print("-" * 50)
    test_chat()
