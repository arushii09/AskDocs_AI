from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

chat_history=[] #list to store msgs

system_message = SystemMessage(content="Your are a helpful AI assistant.")
chat_history.append(system_message)  #adding system msg in chat history

#chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  #add user msg

    #groq using history
    result = model.invoke(chat_history)
    response = result.content   #giving the correct response
    chat_history.append(AIMessage(content=response))   #add ai msg

    print(f"AI: {response}")

print("---Message History---")
print(chat_history)