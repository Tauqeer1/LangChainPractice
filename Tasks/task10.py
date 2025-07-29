from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from typing import Dict, Optional, List


"""
Task: Recipe Ingredient Extractor

Goal: Given a block of text describing a recipe, 
extract a structured list of ingredients 
(e.g., name, quantity, unit).

What to Do: Craft a prompt to guide the LLM to identify 
ingredients. Use a PydanticOutputParser or 
StructuredOutputParser to define the exact schema
for your ingredient list.
"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    quantity: str = Field(description="Quantity of ingredient")
    unit: Optional[str] = None
    
class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    ingredients: List[Ingredient] = Field(description="List of all ingredients")
    
# Initialize the model 
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# Initialize the pydantic parser
parser = PydanticOutputParser(pydantic_object=Recipe)


prompt_text = """
You are an expert recipe ingredient extractor. 
Your task is to carefully read the provided recipe text 
and extract a structured list of ingredients. 
For each recipe, you must identify its title, ingredient list, and
for each ingredient you must identify its name, quantity and unit.
\n\n
**Output Format:**
You **must** respond with a JSON object with title and ingredients list. 
Inside each object must have a ingredient list having
name, quantity and unit in the array represents a single ingredient 
and **must** adhere to the following Pydantic schema:
\n\n

```json

{
    "title": "string",
    "ingredients": [
        {
            "name": "string",
            "quantity": "string",
            "unit": "string"
        }
    ]
}

{recipe_text}
"""

# Create a prompt template
prompt_template = PromptTemplate.from_template(template=prompt_text)

chain = prompt_template | model | parser


result = chain.invoke({
    "recipe_text": "In a bowl, combine 300g bread flour with 5g active dry yeast and 1 tsp sugar. Add 200ml warm water gradually."
})

print("result: ", result)