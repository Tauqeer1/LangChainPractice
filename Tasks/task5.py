from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

"""
Task: Create a simple PromptTemplate.

Description: Define a PromptTemplate with a single input variable (e.g., topic).

Objective: Understand the basic syntax for creating a PromptTemplate.
"""

# Creating the instance of OllamaLLM class
llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")

# There are mainly two common ways of creating Prompt Template

# First way of creating Prompt Template using from_template method 

prompt_template = PromptTemplate.from_template(template="Define in two lines about: {topic}")

# print("prompt_template: ", prompt_template.template)

prompt_template_value = prompt_template.format(topic="Car")


# print('prompt_template_value: ', prompt_template_value)

result = llm.invoke(prompt_template_value)

# print("result: ", result)


# Second way of creating Prompt Template using constructor

prompt_template2 = PromptTemplate(template="Define in two lines about: {topic}",
                                  input_variables=["topic"])

print("prompt_template2: ", prompt_template2)

prompt_template_value2 = prompt_template2.format(topic="China")

print("prompt_template_value2: ", prompt_template_value2)

result2 = llm.invoke(prompt_template_value2)

print("result2: ", result2)


