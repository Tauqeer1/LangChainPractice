from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# 1- Initialize the parser
parser = StrOutputParser()

# 2- Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# 3- Create the Prompt template
prompt_template = PromptTemplate.from_template(template="Generate a short motivational quote about {topic}.")

# # 4- Fill the prompt template value
# prompt_template_value = prompt_template.format(topic="perseverance")

# # 5- Invoke the model
# response = model.invoke(prompt_template_value)

# print(response)

# # 6- Parse the output
# parsed_response = parser.invoke(response)

# print(parsed_response)


chain = prompt_template | model | parser
response = chain.invoke({"topic": "perseverance"})


print(response)

chain.get_graph().print_ascii()