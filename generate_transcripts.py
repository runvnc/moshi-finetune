import os
import json
import google.generativeai as genai

def main():
    # Configure API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        print("Example: export GEMINI_API_KEY='your_api_key_here'")
        return

    genai.configure(api_key=api_key)

    prompt = """
    You are an expert dialogue writer. Generate 50 unique, short dialogue transcripts of an outbound call.
    
    Scenario:
    - Agent Name: Alexis Kim
    - Company: Directory Services
    - Purpose: Calling various businesses to verify pre-employment background checks for different candidates.
    
    Format Requirements:
    - Speaker A is the User (the person answering the phone at the business).
    - Speaker B is the Agent (Alexis Kim).
    - CRITICAL: Speaker A MUST speak first in every single conversation. The conversation must always start with the User answering the phone (e.g., "Hello?", "Acme Corp, how can I help you?").
    - Speaker B (the Agent) speaks second, introducing themselves and stating the purpose of the call.
    - Each conversation should be approximately 6 to 8 turns long, covering the greeting, the initial reason for calling, and a brief back-and-forth exchange.
    - Output strictly as a JSON array of conversations. Each conversation is an array of turn objects.
    
    Example Output:
    [
      [
        {"speaker": "A", "text": "Acme Corporation, how can I direct your call?"},
        {"speaker": "B", "text": "Hi, this is Alexis Kim calling from Directory Services. I'm reaching out to verify employment history for a former employee of yours."}
      ],
      [
        {"speaker": "A", "text": "Hello, Human Resources."},
        {"speaker": "B", "text": "Good morning, my name is Alexis Kim with Directory Services. I'm calling regarding a background check verification for Jane Doe."}
      ]
    ]
    
    Generate exactly 50 conversations in this exact JSON format.
    """

    print("Generating transcripts via Gemini API... (This may take a minute)")
    
    # Using a Flash model for fast, cost-effective generation
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
            )
        )
        
        data = json.loads(response.text)
        print(f"Successfully generated {len(data)} transcripts.")
        
        output_file = "data/outbound/raw_transcripts.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print("Failed to generate or parse transcripts:", e)
        if 'response' in locals():
            print("Raw response:", response.text)

if __name__ == "__main__":
    main()
