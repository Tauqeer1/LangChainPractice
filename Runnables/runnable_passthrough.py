from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the parser
parser = StrOutputParser()

# Create the prompt template
prompt_template = PromptTemplate.from_template(template="Answer the following question: {question}")


runnable = RunnableParallel(
    question=RunnablePassthrough(),
    answer=prompt_template | model | parser
)

result = runnable.invoke({"question": "What's the capital of Pakistan"})

print("result", result)

