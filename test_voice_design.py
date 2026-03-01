import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

print("Designing voice...")
response = client.text_to_voice.design(
    voice_description="A young female with a British accent",
    text="Hello there! This is a test of the voice design system. I am speaking for at least one hundred characters to satisfy the API requirements. I hope this works!"
)

gen_id = response.previews[0].generated_voice_id
print(f"Generated ID: {gen_id}")

try:
    print("Trying to use generated ID directly in text_to_dialogue...")
    audio = client.text_to_dialogue.convert(
        inputs=[{"text": "Hello", "voice_id": gen_id}]
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
