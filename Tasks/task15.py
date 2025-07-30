from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

"""
Project Idea: Simple Q&A Chatbot with Document Context
"""

# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b

# Llama model: llama3.2:1b-instruct-q3_K_L

# This is our sample document text
document_text = """
The quick brown fox jumps over the lazy dog. This is a classic pangram, a sentence
that contains every letter of the English alphabet at least once. Pangrams are
often used to display typefaces or to test typing speed. Another famous pangram is
"Pack my box with five dozen liquor jugs." LangChain is a framework designed to
simplify the creation of applications using large language models. It provides
modules for various tasks, including prompt management, model interaction, and
chaining operations.
"""



# initialize the model
model = ChatOllama(model="gemma3:1b-it-q4_K_M", temperature=0)


# create prompt template
prompt_template = PromptTemplate.from_template(
    template=
    """
    You are a knowledgeable and helpful chatbot tasked with answering questions 
    based solely on the provided text document. Your goal is to provide accurate, 
    concise, and relevant answers. If the document does not contain the information 
    needed to answer the question, clearly state that the information is not 
    available in the document. Do not make up information or rely on 
    external knowledge.

    **Document Content:**
    {document}

    **User Question:**
    {question}

    **Instructions:**
    - Read the document carefully and extract the relevant information to answer 
    the question.
    - Provide a clear and concise answer, directly addressing the user's question.
    - If the answer is not explicitly stated in the document but can be reasonably 
    inferred, provide the inferred answer and note that it is an inference.
    - If the document does not contain relevant information, respond with: 
    "The provided document does not contain information to answer this question."
    - Use a professional and friendly tone.

    **Answer:**
    """)

# initialize the parser 
parser = StrOutputParser()


def get_document_context(_):
    return {"document": document_text}

chain = (
    RunnablePassthrough.assign(document=get_document_context) |
    prompt_template |
    model |
    parser
)

# Test with a question that can be answered from the document
question1 = "What is a pangram?"
answer1 = chain.invoke({"question": question1})
print(f"Question: {question1}\nAnswer: {answer1}\n")

# Test with another question
question2 = "Give me an example of a pangram."
answer2 = chain.invoke({"question": question2})
print(f"Question: {question2}\nAnswer: {answer2}\n")

# Test with a question that cannot be answered from the document
question3 = "Who invented the internet?"
answer3 = chain.invoke({"question": question3})
print(f"Question: {question3}\nAnswer: {answer3}\n")

# Test with a question about LangChain
question4 = "What is LangChain designed for?"
answer4 = chain.invoke({"question": question4})
print(f"Question: {question4}\nAnswer: {answer4}\n")

question5 = "Who jumps on the lazy dog?"
answer5 = chain.invoke({"question": question5})
print(f"Question: {question5}\nAnswer: {answer5}\n")
