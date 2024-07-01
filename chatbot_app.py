import streamlit as st
import time
from openai import OpenAI
# Import Elastic Search
from elasticsearch import Elasticsearch


st.write("## This will be a simple chatbot user interface")


# Add the components of RAG here

# Initialize the client 
es_client = Elasticsearch('http://localhost:9200') # This is the port created after running the docker file

# Provide the name of the index
index_name = "course-questions"


######################## Create the RAG functions #####################

# Create a search function for it
def elastic_search(query):
    # Create the query
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"], # Here you are specifying the boosting of each text field
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    # Query the Elastic Search 
    response = es_client.search(index=index_name, body=search_query)

    # Parse the response of elastic search
    results = []
    for hit in response['hits']['hits']:
        results.append(hit['_source'])
    
    return results

# Create the function to build the prompt
def build_prompt(query, search_results):
    # Create the prompt template
    prompt_template = '''
    You are a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT doesn't contain the answer, output NONE.
    
    QUESTION: {question}
    
    CONTEXT:
    {context}
    '''.strip()
    
    # Create the context from the search results
    context = ''
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    # Putting the context and query all together
    prompt = prompt_template.format(question = query, context = context).strip()
    return prompt

# Intialize the client
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

# Create the function to call the LLM model to generate the response
def llm(prompt):
    # Ask the same question to OpenAI but also provide the appropriate context from RAG
    response = client.chat.completions.create(
        model = 'phi3',
        messages = [{'role':'user','content':prompt}]
    )
    # Return the response
    return response.choices[0].message.content


# Update the rag system function
def rag(query):
    # Search the knowledge base
    search_results = elastic_search(query)
    # Prepare the prompt for the model
    prompt = build_prompt(query, search_results)
    # Use the llm to get the response
    answer = llm(prompt)
    return answer

############################### Building the Application ################################

# Create a function to make the the response more chat like
def response_display(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.07)


# Initializing the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.write_stream(response_display(messages['content']))

# Create a user input to arrive query
if prompt := st.chat_input("Say something"):
    # Display assistant response in chat message container
    with st.chat_message('user'):
        st.write_stream(response_display(prompt))
    # Add the message to the chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Add the response for the chat to user prompt
    response = rag(prompt)
    # Display assistant response
    with st.chat_message('assistant'):
        st.write_stream(response_display(response))
    # Add the message to the chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
