from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

#P-1 creating chatPromptTemplate using a template string
template= "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---Prompt from Template---")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)