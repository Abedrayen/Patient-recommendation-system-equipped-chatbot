import transformers
from transformers import pipeline
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.
Here is the conversation history: {context}
Question: {question}
Answer: 
"""

model = OllamaLLM(model="meditron")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def detect_emotion(text):
    emotions = emotion_classifier(text)

    emotion = max(emotions, key=lambda x: x['score'])
    return emotion['label']

def get_response(user_input):
    result = chain.invoke({"context": context, "question": user_input})
    return result

def process_user_input(user_input):
    emotion = detect_emotion(user_input)
    meditron_response = get_response(user_input)
    chatbot_response = meditron_response
    
    if emotion == "anger":
        chatbot_response = "I understand that you're upset. Let's see how I can help you."
    elif emotion == "sadness":
        chatbot_response = "I'm sorry to hear that you're feeling sad. I'm here to help."
    elif emotion == "joy":
        chatbot_response = "That's great to hear! How can I assist you further?"
   

    return meditron_response, chatbot_response

def handleConversation():
    global context
    context = ""
    print("Welcome to Docare chatbot! Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        meditron_response, chatbot_response = process_user_input(user_input)
        print("DoCare (Modified): ", chatbot_response)
        print( meditron_response)
        
        context += f"\nUser: {user_input}\nDocare (Meditron): {meditron_response}\nDocare (Modified): {chatbot_response}\n"

if __name__ == "__main__":
    handleConversation()