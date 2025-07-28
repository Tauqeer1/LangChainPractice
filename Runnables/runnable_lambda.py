from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# Python function to upper case

def to_uppercase(text: str):
    return text.upper()


# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# initialize the parser
parser = StrOutputParser()

# create the prompt template
prompt_template = PromptTemplate.from_template(template="Tell me a fictional story about the {topic}")


# create uppercase runnable
uppercase_runnable = RunnableLambda(to_uppercase)
runnable = prompt_template | model | parser | uppercase_runnable

result = runnable.invoke({"topic": "food"})

print("result: ", result)
