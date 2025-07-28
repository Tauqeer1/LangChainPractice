from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


# 1- Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# 2- Create the prompt template
prompt_template = PromptTemplate.from_template(template="Tell me a concise fact about {topic}")

# 3- Initialize the parser
parser = StrOutputParser()

# Create the runnable sequence
# sequence = RunnableSequence(prompt_template, model, parser)

# Create runnable with LCEL
sequence = prompt_template | model | parser

result = sequence.invoke({"topic": "universe"})

print("result: ", result)