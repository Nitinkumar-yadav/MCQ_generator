import os
import re
import json
 
# To help construct our Chat Messages
from langchain.prompts import ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
 
# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI
 
# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from fastapi import FastAPI
import uvicorn
from pydantic_core import BaseModel, root_validator


app =FastAPI()
load_dotenv()

def extract_pdf(pdf):
        
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            raw_text= ''
            # text=''
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                     raw_text += text                  
                         
            text_splitter = RecursiveCharacterTextSplitter(        
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )
            chunks = text_splitter.split_text(raw_text)
            return chunks
        
def generating_mcq(chunks):
        response_schemas = [
        ResponseSchema(name="Question", description="A multiple choice question generated."),
        ResponseSchema(name="Options", description="Possible choices for the multiple choice question."),
        ResponseSchema(name="Answer", description="Correct answer for the question.")
        ]

        # The parser that will look for the LLM output in my schema and return it back to me
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # The format instructions that LangChain makes. Let's look at them
        format_instructions = output_parser.get_format_instructions()

        # create ChatGPT object
        chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

        prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("""The multiple choice question should be in three level, In first level easy,In second level medium, In third level hard."""),
            HumanMessagePromptTemplate.from_template("""Generate 10 multiple choice questions from it along with the correct answer. \n 
            {format_instructions} \n
            {user_prompt}
            """)  
        ],
        
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
        )

        for chunk in chunks[:10]:
             
            user_query = prompt.format_prompt(user_prompt = chunk)
            user_query_output =chat_model(user_query.to_messages())

            markdown_text = user_query_output.content

            for json_string in markdown_text:
                json_string = re.search(r'```json\n(.*?)```', markdown_text,re.DOTALL).group(1)
                # Convert JSON string to Python list
                python_list = json.loads(f'[{json_string}]'),

            return python_list
    

class  Doc(BaseModel):
     pdf_url: str 

@app.get('/generating_mcq')
def start():
     return "Welcome Team!"

@app.post('/generating_mcq')
async def generate_mcq_api(request:Doc):
     
        res= request.pdf_url

        filename ="iesc108-min.pdf"
        chunks =extract_pdf(filename)

        mcq =generating_mcq(chunks)
        return mcq

if __name__ == "__main__":
     uvicorn.run(app, host="localhost", port=8000)
