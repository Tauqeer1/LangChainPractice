from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

"""
Task: Parse a simple user profile: Name (str), Age (int), City (str).
"""
# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

class UserProfile(BaseModel):
    name: str
    age: int
    city: str


# initialize the model
model = ChatOllama(model="mistral:7b-instruct-q3_K_L")

# initialize the parser
parser = PydanticOutputParser(pydantic_object=UserProfile)


prompt_text = """
Extract the name, age, and city from the following user description and return it as a JSON object: \n
{user_data}
"""

# create prompt template
prompt_template = ChatPromptTemplate.from_messages([
  ("system", "You are an expert at extracting user profile information. Your task is to extract the name, age, and city from the provided user description. You MUST return the extracted information as a JSON object, strictly following this schema:\n{format_instructions}\nDO NOT include any other text, explanations, or conversational elements in your response. Only output the JSON object, and wrap it in markdown code fences (```json ... ```)."),
  ("human", "{user_data}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt_template | model | parser

result = chain.invoke({"user_data": "My name is Tauqeer, I am 30 years old and live in Karachi"})

print("result:", result)
print(type(result))
print("Name: ", result.name)
print("Age: ", result.age)
print("City: ", result.city)

