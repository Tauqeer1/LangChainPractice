from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field, BaseModel


"""
Recipe Ingredient Extractor:

Goal: From a recipe text, extract a list of ingredients and their quantities.

"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L



# initialize the parser
parser = JsonOutputParser()

# Get formatting instructions for the prompt
# format_instructions = parser.get_format_instructions()


# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# create the prompt text
prompt_template_text = """
Extract the ingredients and their quantities from the following recipe text. 
Return the result as a JSON list of objects, 
where each object has 'ingredient_name' (string) and 'quantity' (string) fields.

Example:
Input: "For the cake you will need 2 cups of flour, 1 cup of sugar, and 3 eggs."
Output: [
    {{"ingredient_name": "flour", "quantity": "2 cups"}},
    {{"ingredient_name": "sugar", "quantity": "1 cup"}},
    {{"ingredient_name": "eggs", "quantity": "3"}}
]

Recipe text: {input_text}
"""

prompt_template = PromptTemplate.from_template(template=prompt_template_text)


lcel_chain = prompt_template | model | parser


print("-----------------Responses---------------------")
cake_recipe = """
For the cake you will need 2 cups of flour, 1 cup of sugar, and 3 eggs.
"""
cake_ingredients = lcel_chain.invoke({"input_text": cake_recipe})
print(f"Cake ingredients: {cake_ingredients}")

sauce_recipe = """
For the pasta sauce you will need 2 cups of crushed tomatoes, 
1 tablespoon of olive oil, and 2 cloves of garlic.
"""
sauce_ingredients = lcel_chain.invoke({"input_text": sauce_recipe})
print(f"Sauce ingredients: {sauce_ingredients}")

pancake_recipe = """
For the pancakes you will need 1 cup of flour, 1 cup of milk, and 2 eggs.
"""
pancake_ingredients = lcel_chain.invoke({"input_text": pancake_recipe})
print(f"Pancake ingredients: {pancake_ingredients}")


guacamole_recipe = """
For the guacamole you will need 3 avocados, 1 lime, and 1/4 cup of chopped cilantro.
"""
guacamole_ingredients = lcel_chain.invoke({"input_text": guacamole_recipe})
print(f"Guacamole ingredients: {guacamole_ingredients}")


chicken_recipe = """
For the chicken curry you will need 1 pound of chicken breast, 2 cups of coconut milk,
and 1 tablespoon of curry powder.
"""
chicken_ingredients = lcel_chain.invoke({"input_text": chicken_recipe})
print(f"chicken ingredients: {chicken_ingredients}")


chocolate_recipe = """
For the chocolate chip cookies you will need 2.5 cups of flour, 
1 cup of butter, and 1 cup of chocolate chips.
"""
chocolate_ingredients = lcel_chain.invoke({"input_text": chocolate_recipe})
print(f"chocolate ingredients: {chocolate_ingredients}")




