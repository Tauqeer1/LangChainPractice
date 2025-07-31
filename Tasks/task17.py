from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
Simple Translation Tool (Limited Languages):

Goal: Translate short sentences from one language to another (e.g., English to Spanish).
"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# initialize the parser 
parser = StrOutputParser()

# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")


prompt_template_text = """
Translate the following short English sentences into Spanish accurately, preserving 
the meaning and tone. Ensure the translations are natural, grammatically correct, and 
culturally appropriate. For each sentence, provide the English original and its 
Spanish translation. Also please make sure don't give any other sentence or word other 
than translation
Here are the sentences to translate: 

{text}

"""

# create the prompt template
prompt_template = PromptTemplate.from_template(template=prompt_template_text)


# create the lcel chain
lcel_chain = prompt_template | model | parser

# invoke the chain
result = lcel_chain.invoke({"text": "How are you ?"})

print("result", result)