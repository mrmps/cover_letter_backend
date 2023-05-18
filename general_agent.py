from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import os

app = FastAPI()

# Define the individual chains
OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7)
llmChatGPT3_5 = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7)


# Understanding the Initial Prompt
prompt_template1 = PromptTemplate(
    input_variables=["initial_prompt"],
    template="Given the initial prompt, identify and summarize the task requirement. Here is the prompt: {initial_prompt} Here are the input variables: job_description (TEXT), resume (TEXT)"
)

chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key="task_understanding")

# Transforming Text Inputs into Structured Data
prompt_template2 = PromptTemplate(
    input_variables=["job_description", "resume", "task_understanding"],
    template="{job_description} {resume} Transform these text inputs into YAML, extacting all information relevant to {task_understanding}"
)

chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key="structured_data")

# Generating the Initial Output based on Structured Inputs
prompt_template3 = PromptTemplate(
    input_variables=["task_understanding", "structured_data"],
    template="{task_understanding} {structured_data} Generate the initial output based on these inputs."
)

chain3 = LLMChain(llm=llm, prompt=prompt_template3, output_key="initial_output")

# Revising the Initial Output
prompt_template4 = PromptTemplate(
    input_variables=["task_understanding", "initial_output"],
    template="{task_understanding} {initial_output} Revise the initial output based on the task understanding."
)

chain4 = LLMChain(llm=llm, prompt=prompt_template4, output_key="revised_output")

# Finalizing the Output
prompt_template5 = PromptTemplate(
    input_variables=["task_understanding", "revised_output"],
    template="{task_understanding} {revised_output} Finalize the output based on the revised output and task understanding."
)

chain5 = LLMChain(llm=llm, prompt=prompt_template5, output_key="final_output")

@app.post("/process_task")
async def process_task_endpoint(background_tasks: BackgroundTasks, initial_prompt: str, job_description: str, resume: str):
    # Run the chains
    task_understanding = chain1.run({"initial_prompt": initial_prompt})
    print("task_understanding", task_understanding)
    structured_data = chain2.run({"job_description": job_description, "resume": resume, "task_understanding": task_understanding})
    print("structured_data", structured_data)
    initial_output = chain3.run({"task_understanding": task_understanding, "structured_data": structured_data})
    print("initial_output", initial_output)
    revised_output = chain4.run({"task_understanding": task_understanding, "initial_output": initial_output})
    print("revised_ouptut", revised_output)
    final_output = chain5.run({"task_understanding": task_understanding, "revised_output": revised_output})
    print("final_output", final_output)
    return {"final_output": final_output}
