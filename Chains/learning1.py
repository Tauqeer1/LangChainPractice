from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# Initiate the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

prompt = PromptTemplate.from_template(template="Tell me joke on this topic: {topic}")

prompt_template_value = prompt.format(topic="cricket")

prompt_template_value1 = prompt.invoke({"topic": "cricket"})


print("prompt_template_value: ", prompt_template_value)
print("prompt_template_value type: ", type(prompt_template_value))
print("prompt_template_value1: ", prompt_template_value1)
print("prompt_template_value type1: ", type(prompt_template_value1))


