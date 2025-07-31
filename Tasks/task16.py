from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


"""
Basic Sentiment Analyzer:

Goal: Determine the sentiment (positive, negative, neutral) of a given piece of text.
"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L


# initialize the parser
parser = StrOutputParser()

# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M")


prompt_template_text = """
Classify the sentiment of the following text. Your response 
MUST be one and only one of these words: 'positive', 'negative', or 'neutral'. 
No other words, punctuation, or explanations are allowed.


Text: "{text}"

Sentiment: 

"""

# create the prompt template
prompt_template = PromptTemplate.from_template(template=prompt_template_text)


lcel_chain = prompt_template | model | parser

example_input = "The weather today is windy."
predicted_sentiment = lcel_chain.invoke({"text": example_input})

print(f"Input: '{example_input}'")
print(f"Predicted Sentiment: {predicted_sentiment}")