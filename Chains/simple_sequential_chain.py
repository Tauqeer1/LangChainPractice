from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# 1 - Initialize the parser
parser = StrOutputParser()

# 2 - Initialize the model
model = ChatOllama(model="llama3.2:1b-instruct-q3_K_L")

# 3 - Create the Prompt Template
prompt_template1 = PromptTemplate.from_template(template="Generate a detailed report on {topic}")

prompt_template2 = PromptTemplate.from_template(template="Generate a five pointer summary from the following text \n {text}")

# chain = prompt_template1 | model | parser | prompt_template2 | model | parser

chain1 = prompt_template1 | model | parser

chain2 = prompt_template2 | model | parser

combine_chain = chain1 | chain2

response = combine_chain.invoke({"topic": "Unemployment in Pakistan"})

print("response: ", response)

combine_chain.get_graph().print_ascii()
