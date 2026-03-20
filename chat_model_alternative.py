from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

messages=[
    SystemMessage=("Solve the following problems"),
    HumanMessage=("What is 81 divided by 9?"),
]

#creating groq ai model
model = ChatGroq(model="llama-3.1-8b-instant")

result= model.invoke(messages)
print(f"Answer from Groq: {result.container}")