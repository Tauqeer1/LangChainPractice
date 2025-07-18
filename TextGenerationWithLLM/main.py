import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


hugging_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# this works with mistralai/Magistral-Small-2506

# llm = HuggingFaceEndpoint(
#         repo_id="mistralai/Magistral-Small-2506", 
#         huggingfacehub_api_token=hugging_api_key, 
#         task="text-generation")

# model = ChatHuggingFace(llm=llm)
# result = model.invoke([HumanMessage("What is the capital of Sindh")])
# print(result.content)


# Another model

# 2. Define the Hugging Face model ID
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 3. Create a Langchain ChatHuggingFace model
# ChatHuggingFace internally handles the conversion to the conversational API format.
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7,
    max_new_tokens=256,
    # You might need to specify the task explicitly here depending on the version,
    # but often it's inferred correctly by ChatHuggingFace for chat models.
    # task="conversational" # If it still errors, try adding this.
)

model = ChatHuggingFace(llm=llm)
# 4. Use the Chat Model to generate text
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is the capital of Pakistan?"),
]

response = model.invoke(messages) # Use .invoke() for chat models
print(response)

