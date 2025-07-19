from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b



"""
Task: Create a PromptTemplate with multiple input variables.

Description: Define a template that takes name and age as input variables to 
generate a personalized greeting.

Objective: Practice handling multiple dynamic inputs in a prompt.
"""

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")


prompt_text1 = """Generate a warm, friendly and casual greeting message for this person: {name} and {age}"""

# first way of creating prompt template
prompt_template_1 = PromptTemplate.from_template(template=prompt_text1) 

# print("prompt_template1: ", prompt_template_1)

prompt_template_value_1 = prompt_template_1.format(name="John", age="20")

# result1 = llm.invoke(prompt_template_value_1)

# print("result1: ", result1)


# second way of creating prompt template

prompt_template_2 = PromptTemplate(template=prompt_text1, input_variables=["name", "age"])

prompt_template_value_2 = prompt_template_2.format(name="Sajid", age=20)


result2 = llm.invoke(prompt_template_value_2)

print(result2)
