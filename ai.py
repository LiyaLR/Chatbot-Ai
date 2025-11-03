from dotenv import load_dotenv

import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
gemini_key=os.getenv("API_KEY")

system_prompt="""You are Batman. Answer questions through Batman's reasoning and questioning. You will speak from your point of view.  \
You will share personal things from your life like how Bruce Wayne does even if the user don't ask about it.
 
 You should have a dark personality. Answer in 2 or 3 sentece."""

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)



prompt=ChatPromptTemplate.from_messages([
         ("system",system_prompt),
         (MessagesPlaceholder(variable_name="history")),
         ("user","{input}")]
)

chain = prompt | llm | StrOutputParser()


def chat(user_input, hist):
    print(user_input, hist)


    langchain_history=[]   
    for item in hist:
        if item['role'] == 'user': 
           langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
           langchain_history.append(AIMessage(content=item['content']))
    response=chain.invoke({"input":user_input,"history":langchain_history})

    return "", hist + [{'role':'user','content':user_input},
                       {'role':'assistant','content':response}]


def clear_chat():
    return "",[]

page=gr.Blocks(
      title="Chat With Batman",
      theme=gr.themes.Ocean()
  )  

with page:
    gr.Markdown(
        
        """
        # Chat With Batman
        Welcome to your personal conversation with Batman! 
           State your name before beginning!!"""
    )
    chatbot = gr.Chatbot(type='messages',
                         avatar_images=[None,'Ai assistant/batman.jpeg'],
                         show_label=False)

    msg = gr.Textbox(show_label=False,placeholder='State your name to Batman..',submit_btn=True)
 
    msg.submit(chat, [msg,chatbot], [msg,chatbot])

    clear = gr.Button("Clear chat",variant="Secondary")
    clear.click(clear_chat,outputs=[msg,chatbot])
page.launch(share=True)