from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}.")       ,
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

escalate_feeback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
)

classification_template =ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.,")
    ]
)

#defining runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model |StrOutputParser()
    ),
    escalate_feeback_template | model |StrOutputParser()
)

#classification chain
classification_chain = classification_template | model |StrOutputParser()

chain = classification_chain |branches

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})
print(result)