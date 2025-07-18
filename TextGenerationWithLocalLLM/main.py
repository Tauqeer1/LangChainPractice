# import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='google/flan-t5-small', 
    task='text-generation', 
    pipeline_kwargs={"max_new_tokens": 50, "temperature": 0.7}
)

# model = ChatHuggingFace(llm=llm)

result = llm.invoke("What is the capital of USA")

print(result)
