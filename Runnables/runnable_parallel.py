from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the parser
parser = StrOutputParser()

# Create a prompt template
joke_prompt_template = PromptTemplate.from_template(template="Tell me a short, funny joke about {topic}")
haiku_prompt_template = PromptTemplate.from_template(template="Tell a haiku about {topic}")

# Create runnable sequence using LCEL
joke_sequence = joke_prompt_template | model | parser
haiku_sequence = haiku_prompt_template | model | parser

runnable_parallel = RunnableParallel(joke=joke_sequence, haiku=haiku_sequence)

result = runnable_parallel.invoke({"topic": "cats"})

print("result", result)