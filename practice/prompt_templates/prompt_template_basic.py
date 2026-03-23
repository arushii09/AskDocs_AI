from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

#Part-1 creating chatPromptTemplate using a template string-----
#template= "Tell me a joke about {topic}."
#prompt_template = ChatPromptTemplate.from_template(template)

#print("---Prompt from Template---")
#prompt = prompt_template.invoke({"topic": "cats"})
#print(prompt)

#Part-2 prompt with multiple values--------
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)

# PART 3: Prompt with System and Human Messages (Using Tuples)
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     ("human", "Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)


# # Extra Informoation about Part 3.
# # This does work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me 3 jokes."),                          NO TUPLE IN HUMMANMSG
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)


# This does NOT work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),                #{JOKE_COUNT} DOESN'T WORK HERE
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
