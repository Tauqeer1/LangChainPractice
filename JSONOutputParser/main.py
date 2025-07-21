from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser



# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


# 1 - Initialize the LLM (Create the ChatOllama instance)
llm = ChatOllama(model="llama3.2:1b-instruct-q3_K_L")

# 2 - Initialize the json output parser
parser = JsonOutputParser()

# 3 - Create a Prompt template that gives format instruction
prompt_template = PromptTemplate.from_template(
    template="Answer the user query. \n{format_instructions}\n{query}", 
    partial_variables={"format_instructions": parser.get_format_instructions()})

# 4 - Fill the Prompt Template with Values
prompt_template_value = prompt_template.format(query="Tell me about the movie 'Inspection'. I need it's director and main stars.")

# 5 - Invoke the llm and get's the result
result = llm.invoke(prompt_template_value)

# print("result: ", result)
# print("type of result: ", type(result))

# 6 - Parsed the LLM output to a JSON output 
json_parsed_result = parser.invoke(result)

print("Parsed result: ", json_parsed_result)
print("type of parsed result: ", type(json_parsed_result))


print(json_parsed_result.get('director'))