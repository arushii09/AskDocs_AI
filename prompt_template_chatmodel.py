from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

#PART-1 creating a chatPromptTemplate using template string
print("---Prompt from template---")
template= "tell me a joke about {topic}."  #string template
prompt_template= ChatPromptTemplate.from_template(template)  #calling ChatPromptTemplate make an actual template from string tempalte (easy for langchain to manipulate it)

prompt = prompt_template.invoke({"topic": "cats"})  #invoke to taking that string and replacing the values
result = model.invoke(prompt)
print(result.content)

#PART-2 prompt with multiple placeholders
print("\n----- Prompt with Multiple Placeholders -----\n")
template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
result=model.invoke(prompt)
print(result.content)

#PART-3 prompt with system and human msg (using tuples)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)