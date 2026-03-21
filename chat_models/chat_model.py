from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

response = model.invoke("What is 81 divided by 9?")

print(response.content)