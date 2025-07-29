from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

"""
Task: Parse a book's metadata: Title (str), Author (str), Publication Year (int), Genre (str).
"""
# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# create pydantic model
class Book(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author of the book")
    publication_year: int = Field(description="Year when the book published")
    genre: str = Field(description="Book Genre")
    
# initialize the parser
parser = PydanticOutputParser(pydantic_object=Book)

# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")

# create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        You are an expert data extraction assistant. Your task is to analyze the 
        provided text and extract structured information about a product in JSON format. 
        Follow these instructions:
        - Extract the title, author, publication_year and genre from the text.
        - If the title is missing, use 'Not available' and include a note in the JSON output as a comment.
        - If the author is missing, use 'Not available' and include a note in the JSON output as a comment.
        - If the publication_year is missing, use 'Not available' and include a note in the JSON output as a comment.
        - If the genre is missing, use 'Not available' and include a note in the JSON output as a comment.
        - Ensure the output strictly adheres to the JSON schema provided below.
        - Do not include any additional fields beyond those specified.
        **Sample JSON Output**:
        ```json
        {{
            "title": "Example title",
            "author": "Author name",
            "publication_year": 1990,
            "genre": "Comic"
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
The Great Adventure, by Jane Smith, published in 2020. It's a fantasy novel.
"""
print("Response 1: ", chain.invoke({"input_text": input_prompt_text1}))
input_prompt_text2 = """
Discover 'The Starlit Path' by Jane Ellison, published in 2023. This captivating 
fantasy novel weaves a tale of magic and adventure
"""
print("Response 2: ", chain.invoke({"input_text": input_prompt_text2}))

input_prompt_text3 = """
Read 'Echoes of Time', a thrilling science fiction story by Mark Rivera, released in 2019.
"""
print("Response 3: ", chain.invoke({"input_text": input_prompt_text3}))

input_prompt_text4 = """
Whispers in the Dark' is a mystery novel by Sarah Lin, first published in 2021, 
perfect for fans of suspenseful storytelling
"""
print("Response 4: ", chain.invoke({"input_text": input_prompt_text4}))