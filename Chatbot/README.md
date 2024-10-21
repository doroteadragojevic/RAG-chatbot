#Random Fortune Telling Bot

This repository contains the Random Fortune Telling Bot, a simple chatbot developed using Langchain, Pinecone, and Hugging Face models. The bot uses a Retrieval-Augmented Generation (RAG) approach to provide answers to user questions by referencing a built-in set of documents as context.

##Repository Contents:
main.py: The main script that defines the Chatbot class, utilizing Langchain and Pinecone for document processing and retrieval.
frontend.py: A Streamlit application that allows users to interact with the bot through a simple web interface.
knowledge-db.txt: A text file containing documents the bot uses as the basis for its responses.
requirements.txt: A list of Python dependencies required to run the project.

##How to Use
Clone the repository.
Install the required packages by running pip install -r requirements.txt.
Run the application using streamlit run frontend.py.

##About the Project
This project was created as part of a presentation at the Faculty of Electrical Engineering and Computing (FER) in Zagreb. The goal was to demonstrate impact of RAG in chatbots. 