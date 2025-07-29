from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

"""
Task: Extract a product description: Name (str), Price (float), Currency (str), Description (str).
"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# create the pydantic object
class Product(BaseModel):
    name: str = Field(description="Name of the product")
    price: float = Field(description="Price of the product")
    currency: str = Field(description="Currency in which product price is present")
    description: str = Field(description="Description of the product")


# Initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=Product)

# Create the prompt
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        You are an expert data extraction assistant. Your task is to analyze the 
        provided text and extract structured information about a product in JSON format. 
        Follow these instructions:
        - Extract the product name, price, currency (USD, CAD, SAR, PKR), and description from the text.
        - If the price is missing, use 0.0 and include a note in the JSON output as a comment.
        - If the currency is not specified, use 'USD' and include a note in the JSON output as a comment.
        - If the description is missing, use '' and include a note in the JSON output as a comment.
        - Ensure the output strictly adheres to the JSON schema provided below.
        - Do not include any additional fields beyond those specified.
        **Sample JSON Output**:
        ```json
        {{
            "name": "Example Product",
            "price": 49.99,
            "currency": "USD",
            "description": "This is the description of the product"
        }}
        ```
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
         **Input Text**:
        {input_text}

        **Output**:
        Please provide the extracted information in the specified JSON format.
        """
    )
])

chain = prompt_template | model | parser



print("------------------Responses-----------------------")
input_prompt_text1 = """
Introducing the EcoGlow Lamp, available for €79.50 EUR. This energy-efficient LED lamp provides 
adjustable brightness and a modern minimalist design, ideal for any home or office space
"""
print("Response 1: ", chain.invoke({"input_text": input_prompt_text1}))
input_prompt_text2 = """
For only £129.99 GBP, get the PowerFit Treadmill! Compact and foldable, 
this treadmill features a digital display, 12 preset programs, and a quiet motor for home workouts.
"""
print("Response 2: ", chain.invoke({"input_text": input_prompt_text2}))

input_prompt_text3 = """
The NoiseBuster Headphones are now on sale for $249.95 CAD. Enjoy premium sound quality, 
active noise cancellation, and 20 hours of wireless playback for an immersive audio experience.
"""
print("Response 3: ", chain.invoke({"input_text": input_prompt_text3}))