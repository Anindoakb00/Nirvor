import os
from dotenv import load_dotenv
from openai import OpenAI

# Load your .env file where BASETEN_API_KEY is stored
load_dotenv()

# Initialize the OpenAI client pointing to Baseten's hackathon endpoint
client = OpenAI(
    base_url=os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
    api_key=os.getenv("BASETEN_API_KEY"),
)

print("Testing the API")

# Call the model
response = client.chat.completions.create(
    model="meta-models/muse-spark-open-1026",
    messages=[
        {
            "role": "user",
            "content": (
                "Hello. Confirm you are working by replying with 'Baseten API"
                " is fully operational'."
            ),
        }
    ],
    temperature=0.1,
)

# Print the model's reply
print("\nResponse from Baseten API:")
print(response.choices[0].message.content)
