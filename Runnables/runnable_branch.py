from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the parser
parser = StrOutputParser()


"""Condition 1: Check if the input contains "product" or "describe"""
def is_product_query(input_text: str):
    text = input_text.lower()
    return "product" in text or "describe" in text or "tell me about" in text

product_condition = RunnableLambda(is_product_query)

"""Condition 2: Check if the input contains common math terms or operators"""
def is_math_query(input_text: str):
    text = input_text.lower()
    return any(op in text for op in ["what is", "calculate", "sum", "difference", "plus", "minus", "times", "divided by", "+", "-", "*", "/"])


math_condition = RunnableLambda(is_math_query)


# Fallback Condition: For anything else
fallback_condition = RunnableLambda(lambda x: True)


# Create Product Template
product_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant specialized in providing concise and helpful product descriptions. Focus on key features and benefits."),
    ("human", "{query}")
])

# Create Product Runnable sequence
product_branch = { "query": RunnablePassthrough() } | product_template | model | parser


# Create Math Problem Template
math_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant specialized in solving simple mathematical questions. Provide the answer directly."),
    ("human", "{query}")
])

# Create math runnable sequence
math_branch = { "query": RunnablePassthrough() } | math_template | model | parser


# unhandled query branch
unhandled_query_branch = RunnableLambda(lambda x: "I can only help with product descriptions or simple math questions. Please refine your query.")


# Create runnable branch
runnable_branch = RunnableBranch(
    (product_condition, product_branch),
    (math_condition, math_branch),
    (unhandled_query_branch)
)

# --- Test the runnable branch ---

print("--- Testing Product Query ---")
query1 = "Can you describe the new 'Quantum Leap' smartwatch?"
print(f"Input: '{query1}'")
print(f"Response: {runnable_branch.invoke(query1)}\n")

print("--- Testing Math Query ---")
query2 = "What is 150 plus 75?"
print(f"Input: '{query2}'")
print(f"Response: {runnable_branch.invoke(query2)}\n")

print("--- Testing Another Product Query ---")
query3 = "Tell me about the features of the 'EcoFlow' solar panel."
print(f"Input: '{query3}'")
print(f"Response: {runnable_branch.invoke(query3)}\n")

print("--- Testing Another Math Query ---")
query4 = "Calculate 25 times 12."
print(f"Input: '{query4}'")
print(f"Response: {runnable_branch.invoke(query4)}\n")

print("--- Testing Unhandled Query ---")
query5 = "What is the weather like today in London?"
print(f"Input: '{query5}'")
print(f"Response: {runnable_branch.invoke(query5)}\n")

