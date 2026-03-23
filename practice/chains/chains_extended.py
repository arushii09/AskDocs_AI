from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

#defining additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

#combined chain using LCEL
chain= prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "doctors", "joke_count": 2})
print(result)