from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# JSONOutputParser with Pydantic for schema definition and validation

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# 1 - Define the desired data structure using pydantic base model
class MovieInfo(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    release_year: int = Field(description="The year the movie was released")
    main_actors: List[str] = Field(description="A list of main actors in the movie")
    genre: str = Field(description="The primary genre of the movie")

# 2 - Initialize the llm (create the ChatOllama instance)
llm = ChatOllama(model="mistral:7b-instruct-q3_K_L")

# 3 - Initialize the JSONOutputParser with your Pydantic Object
parser = JsonOutputParser(pydantic_object=MovieInfo)

# 4 - Create a PromptTemplate
prompt_template = PromptTemplate.from_template(
    template="Answer the user query.\n{format_instructions}\n{query}",
    partial_variables={"format_instructions": parser.get_format_instructions()})

# 5 - Fill the prompt template with values
prompt_template_value = prompt_template.format(query="Give me detailed information about the movie 'Dune' (2021).")

# 6 - Invoke the llm
result = llm.invoke(prompt_template_value)

print("result: ", result.content)
print("type: ", type(result))

# 7 - Parse the result to json format
parsed_result = parser.invoke(result)

print("parsed_result: ", parsed_result)
print("type of parsed result: ", type(parsed_result))

# Values from the parsed result
print("Title: ", parsed_result.get('title'))
print("Director: ", parsed_result.get('director'))
print("Release year: ", parsed_result.get('release_year'))
print("Main Actors:", parsed_result.get('main_actors'))
print("Genre", parsed_result.get('genre'))
