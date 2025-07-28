from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

"""
Task: The Formal Email Responder

Goal: Generate a polite and formal email response to a customer inquiry, 
based on a provided query and desired action (e.g., "confirm receipt," "request more info," "apologize").

What to Do: Design a prompt that takes the customer inquiry and the 
desired action as input. Consider using an OutputParser to ensure the output is 
just the email body, without extra conversational text.
"""


# Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the parser
parser = StrOutputParser()

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an AI assistant specialized in drafting formal and polite email responses to customer inquiries."
            "Your task is to generate only the body of an email, without any salutations, closings, or subject lines."
        ),
        HumanMessagePromptTemplate.from_template(
            "Here's the customer's original inquiry:\n\"{customer_inquiry}\"\n\n"
            "Based on the inquiry, your desired action for this response is: \"{desired_action}\".\n\n"
            "Draft a formal and polite email body that fulfills the desired action."
        ),
    ]
)

chain = chat_prompt_template | model | parser

result = chain.invoke({
    "customer_inquiry": "I am writing to inquire about the status of my recent order, #12345. It was placed on July 20, 2025, and I haven't received any updates.", 
    "desired_action": "provide an update on their order status"})


print("result: ", result);