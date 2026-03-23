from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes"),
    ]
)

#creating individual runnables
format_prompt= RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model= RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output= RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "lawyers", "joke_count": 2})

print(response)