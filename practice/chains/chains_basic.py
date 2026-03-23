from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

#defining prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

#creating combine chain using LCEL
chain = prompt_template | model | StrOutputParser()  #StrOutputParser -> spit everything out to a string parser

#running the chain
result = chain.invoke({"topic": "engineer", "joke_count": 3})

print(result)