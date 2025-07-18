from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaLLM

# model: gemma3:1b-it-q4_K_M

llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")

"""
Task 1: Simple Question

Objective: Get the LLM to answer a straightforward factual question.

Prompt Requirement: Create a prompt that asks "What is the capital of France?"

Expected LLM Output (example): "The capital of France is Paris."
"""

prompt1 = PromptTemplate.from_template('What is the capital of France?')

# print(f"Prompt1: {prompt1}")

formatted_prompt1 = prompt1.format()

# print(f"Formatted Prompt1: {formatted_prompt1}")

# response1 = llm.invoke(formatted_prompt1)

# print(f"Response1: {response1}")




"""
Task 2: Targeted Information Extraction

Objective: Extract specific information from a given text.

Scenario: You have the following text: 
"The quick brown fox jumps over the lazy dog. This sentence is a classic pangram, 
often used to display fonts because it contains every letter of the alphabet."

Prompt Requirement: Create a prompt that asks the LLM to identify all the animals mentioned in the text.

Expected LLM Output (example): "The animals mentioned are: fox, dog.
"""

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that extracts information from text."),
    ("user", "Identify all the animals from the following text: {text}")
])

# print(f"Prompt2: {prompt2}")

formatted_prompt2 = prompt2.format(text="The quick brown fox jumps over the lazy dog")

# print(f"Formatted Prompt: {formatted_prompt2}")

# response2 = llm.invoke(formatted_prompt2)

# print(f"Response 2: {response2}")

'''
Task 3: Summarization

Objective: Condense a longer piece of text into a concise summary.

Scenario: You have the following text: "Artificial intelligence (AI) is a rapidly expanding 
field focused on creating machines that can think, learn, and act like humans. Its applications 
range from self-driving cars and medical diagnosis to natural language processing and creative arts. 
While AI holds immense promise for solving complex problems, ethical considerations 
regarding bias, privacy, and job displacement are also significant areas of discussion."

Prompt Requirement: Create a prompt that instructs the LLM to summarize the provided 
text in no more than two sentences.

Expected LLM Output (example): "Artificial intelligence (AI) is a growing field dedicated to 
developing machines with human-like cognitive abilities, with diverse applications. Despite its 
potential, it raises important ethical concerns about bias, privacy, and job displacement.
'''


prompt_text3 = """Artificial intelligence (AI) is a rapidly expanding 
field focused on creating machines that can think, learn, and act like humans. Its applications 
range from self-driving cars and medical diagnosis to natural language processing and creative arts. 
While AI holds immense promise for solving complex problems, ethical considerations 
regarding bias, privacy, and job displacement are also significant areas of discussion."""

prompt3 = PromptTemplate.from_template("Summarize the provided text in no more than two sentences: {text}")

formatted_prompt3 = prompt3.format(text=prompt_text3)

# response3 = llm.invoke(formatted_prompt3)

# print(f"Response3: {response3}")



"""
Task 4: Tone and Style Adjustment

Objective: Guide the LLM to rewrite a sentence or paragraph in a specific tone or style.

Scenario: You have the following sentence: "The new software update has fixed several critical 
bugs and improved overall performance."

Prompt Requirement: Create a prompt that asks the LLM to rephrase this sentence in a more formal 
and sophisticated tone.

Expected LLM Output (example): "The recent software update has successfully rectified several 
critical anomalies and enhanced the holistic operational efficiency.
"""

prompt_text4 = """The new software update has fixed several critical
                    bugs and improved overall performance."""

# prompt4 = PromptTemplate.from_template("Please write the sentence in a more formal and sophisticated tone: {text}")

# formatted_prompt4 = prompt4.format(text=prompt_text4)

# response4 = llm.invoke(formatted_prompt4)

# print(f"Response4: {response4}")


template = PromptTemplate(template="Please write the sentence in a more formal and sophisticated tone: {text}", 
                          input_variables=['text'])


prompt_template = template.invoke({'text': prompt_text4})

print(f"Prompt template: {prompt_template}")

response = llm.invoke(prompt_template)

print(response)



