from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

#system msgs comes 1st as of the context of the conversation
messages=[
    SystemMessage(content="Solve the following math problems"), 
    HumanMessage(content="What is 81 divided by9?"),
]

result=model.invoke(messages)
#print(f"Answer from AI: {result.content}")

messages=[
    SystemMessage(content="Solve the following math problems"), 
    HumanMessage(content="What is 81 divided by9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]
result=model.invoke(messages)
print(f"Answer from AI: {result.content}")