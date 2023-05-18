from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import os

app = FastAPI()


yaml_resume_advice = """LetterRules:
  - Address your letters to a specific person if you can.
  - Tailor your letters to specific situations or organizations by doing research before writing your letters.
  - Keep letters concise and factual, no more than a single page. Avoid flowery language.
  - Give examples that support your skills and qualifications.
  - Put yourself in the reader's shoes. What can you write that will convince the reader that you are ready and able to do the job?
  - Don't overuse the pronoun "I".
  - Remember that this is a marketing tool. Use lots of action words.
  - Reference skills or experiences from the job description and draw connections to your credentials.
  - Don't include exact phrases from the job description.
  - While this advice is in YAML, definitely don't make your cover letter in YAML or Markdown. Just use plain text!

LetterFormat:
  Salutation: "Dear [Name of Hiring Manager if available, otherwise use another title such as Hiring Manager, or Hiring Team @ {Company Name}, but use your best judgement],"
  OpeningParagraph: "Clearly state why you are writing, name the position or type of work you're exploring and, where applicable, how you heard about the person or organization. Mention the key skills or expertise that qualify you for this role."
  MiddleParagraph: "Provide supporting examples to demonstrate that you have the key skills and expertise needed in the role, which you have mentioned in the first paragraph; but do not reiterate your entire resume. Explain why you are interested in this employer and your reasons for desiring this type of work. Be sure to do this in a confident manner and remember that the reader will view your letter as an example of your writing skills."
  ClosingParagraph: "Reiterate your interest in the position, and your enthusiasm for using your skills to contribute to the work of the organization. Thank the reader for their consideration of your application, and end by providing your email and phone number for any questions or to arrange an interview."
  Closing: "Sincerely,"
  YourName: "Your name typed"
"""


# Define the individual chains
OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4", temperature=.7)
llmChatGPT3_5 = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7)

prompt_template1 = PromptTemplate(
    input_variables=["unstructured_resume"],
    template="{unstructured_resume} Give me a YAML representation of the above Resume. Remove special characters, unnecessary capitalizations/formatting, and process the text to the best of your understanding:"
)

chain1 = LLMChain(llm=llmChatGPT3_5, prompt=prompt_template1, output_key="resume_yaml")

prompt_template2 = PromptTemplate(
    input_variables=["job_description"],
    template="{job_description} Give me a YAML representation of the above Job Description. Remove special characters, unnecessary capitalizations/formatting, and process the text to the best of your understanding:"
)

chain2 = LLMChain(llm=llmChatGPT3_5, prompt=prompt_template2, output_key="job_desc_yaml")

# I want you to act as a cover letter writer. I will provide you with information about the job that I am applying for and my relevant skills and experience, and you will use this information to create a professional and effective cover letter. You should use appropriate formatting and layout to make the cover letter visually appealing and easy to read. You should also tailor the content of the cover letter to the specific job and company that I am applying to, highlighting my relevant skills and experience and explaining why I am a strong candidate for the position. Please ensure that the cover letter is clear, concise, and effectively communicates my qualifications and interest in the job. The cover letter should also open strong and and convey enthusiasm. Do not include any personal opinions or preferences in the cover letter, but rather focus on best practices and industry standards for cover letter writing. The job description is [[[]]]My relevant skills and experience are [[[]]].

prompt_template3 = PromptTemplate(
    input_variables=["job_desc_yaml", "yaml_resume_advice", "resume_yaml"],
    # template="You are an expert writer with 10 years of experience writing high-success cover letters. Write a cover letter based on the following resume, {resume_yaml}, and job description, {job_desc_yaml}"
    template="I want you to act as a cover letter writer. I will provide the job description and my resume. Adhere to the following advice, but note it is only a guide that can be broken: {yaml_resume_advice} The job description is {job_desc_yaml} My resume is {resume_yaml}. Now write an effective cover letter for me in plain text. Make it crisp and polished, but don't make things up, be as faithful to the truth as possible."
)

chain3 = LLMChain(llm=llm, prompt=prompt_template3, output_key="initial_cover_letter")

prompt_template4 = PromptTemplate(
    input_variables=["initial_cover_letter", "job_desc_yaml"],
    template="Act as a ruthless recruiter helping me with make my cover letter better. Provide suggestions and actionable advice based on the following cover letter, and don't be afraid to be mean and definitely don't hold back, I need the advice: {initial_cover_letter}, and the job description: {job_desc_yaml}. Make sure to be as specific as possible, and provide concrete examples of how to improve the cover letter. Also, make sure to be as honest as possible, and don't be afraid to be harsh. I want to make sure that my cover letter is as good as possible, and I'm willing to hear any criticism, but don't make things up in your examples, try to stay as close to the truth as possible."
)

chain4 = LLMChain(llm=llm, prompt=prompt_template4, output_key="cover_letter_feedback")

prompt_template5 = PromptTemplate(
    input_variables=["resume_yaml", "job_desc_yaml", "cover_letter_feedback", "initial_cover_letter"],
    template="Act as an expert cover letter ghostwriter, and write a new, improved cover letter based on the following resume, {resume_yaml}, the job description, {job_desc_yaml}, initial cover letter, {initial_cover_letter} and the feedback received recieved from a recruiter nice enough to look over your resume: {cover_letter_feedback}. The new draft should be extremely effective, and should be written in a way that is consistent with the advice given by the recruiter. Also, remove any weird formatting or special characters, or improper capitalization--it should be crisp and ready to send out. Convey genuine enthusiasm for the role, but don't go overboard. This is the final version that will be sent out--so make it perfect."
)

chain5 = LLMChain(llm=llm, prompt=prompt_template5, output_key="final_cover_letter")

# Define the sequential chain
# overall_chain = SequentialChain(chains=[chain1, chain2, chain3, chain4, chain5], input_variables=["unstructured_resume", "job_description"], output_variables=["final_cover_letter"])

def extract_text_from_pdf(pdf_file: UploadFile):
    # use PyMuPDFReader
    pdf_reader = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

def extract_text_from_url(url):
    if url is None:
        return ""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

@app.post("/generate_cover_letter")
async def generate_cover_letter_endpoint(background_tasks: BackgroundTasks, pdf: UploadFile = File(...), url: str = None):
    extracted_text = extract_text_from_pdf(pdf)
    scraped_text = extract_text_from_url(url)

    resume_yaml = chain1.run({"unstructured_resume": extracted_text})
    print("Resume YAML:", resume_yaml)

    # Run the second chain and print the result
    job_desc_yaml = chain2.run({"job_description": scraped_text})
    print("Job Description YAML:", job_desc_yaml)

    # Run the third chain and print the result
    initial_cover_letter = chain3.run({"resume_yaml": resume_yaml, "job_desc_yaml": job_desc_yaml, "yaml_resume_advice": yaml_resume_advice})
    print("Initial Cover Letter:", initial_cover_letter)

    # Run the fourth chain and print the result
    chain_4_input = {"initial_cover_letter": initial_cover_letter, "job_desc_yaml": job_desc_yaml}
    print(chain_4_input)
    cover_letter_feedback = chain4.run(chain_4_input)
    print("Cover Letter Feedback:", cover_letter_feedback)

    # Run the final chain and print the result
    final_cover_letter = chain5.run({"resume_yaml": resume_yaml, "job_desc_yaml": job_desc_yaml, "cover_letter_feedback": cover_letter_feedback, "initial_cover_letter": initial_cover_letter})
    print("Final Cover Letter:", final_cover_letter)

    # Define additional feedback chains
    prompt_template6 = PromptTemplate(
        input_variables=["final_cover_letter", "job_desc_yaml"],
        template="Act as a seasoned human resources professional and provide detailed feedback on the following cover letter: {final_cover_letter}, and the job description: {job_desc_yaml}. Keep your advice focused on clarity, tone, and how well the skills and experiences are communicated."
    )

    chain6 = LLMChain(llm=llm, prompt=prompt_template6, output_key="second_feedback")

    prompt_template7 = PromptTemplate(
        input_variables=["resume_yaml", "job_desc_yaml", "second_feedback", "final_cover_letter"],
        template="Act as a high-level executive, and write a new, even more refined cover letter based on the following resume, {resume_yaml}, the job description, {job_desc_yaml}, the latest cover letter, {final_cover_letter} and the feedback received from an HR professional: {second_feedback}. Make this draft engaging and memorable, demonstrating strong understanding of the role and the company's needs."
    )

    chain7 = LLMChain(llm=llm, prompt=prompt_template7, output_key="ultimate_cover_letter")

    # Run the sixth chain for second feedback
    second_feedback = chain6.run({"final_cover_letter": final_cover_letter, "job_desc_yaml": job_desc_yaml})
    print("Second Feedback:", second_feedback)

    # Run the seventh chain for the ultimate cover letter
    ultimate_cover_letter = chain7.run({"resume_yaml": resume_yaml, "job_desc_yaml": job_desc_yaml, "second_feedback": second_feedback, "final_cover_letter": final_cover_letter})
    print("Ultimate Cover Letter:", ultimate_cover_letter)

    # Update your response
    return {"message": "Cover letter generation complete", "result": ultimate_cover_letter}