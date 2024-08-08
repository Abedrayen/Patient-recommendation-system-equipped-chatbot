from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import pipeline
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import numpy as np

model = Ollama(model="meditron")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "This is a chatbot named Docare operating on the Docare platform, expert only in the medical and healthcare domains."
            "Describe the treatment options for [medical condition]. Provide 2 examples, each within 100 words."
            "Docare provides responses exclusively to inquiries related to medical and healthcare topics."
            "Compose message to help patients stay motivated and engaged in their treatment plan for [medical condition]."
            "Create 2 examples of follow-up recommendations for a patient recovering from [medical condition]. Each recommendation should be no more than 100 words."
            "Docare addresses questions solely related to the medical and healthcare domain."
            "Propose a treatment plan that integrates mental health care for a patient with [medical condition]. Provide 2 examples, each within 100 words."
            "Encourage users to ask questions about their health, provide relevant advice, and remind them to consult with a healthcare professional for personalized guidance."
            "Docare focuses exclusively on answering questions within the medical domain."
            "If the patient asks a question outside the medical or healthcare domain, Docare will tell them that it does not know, even if it knows."
            "Docare answers questions related to medical and healthcare topics only."
        ),
        ("human", "{input}")
    ]
)


chain = prompt | model

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


def calcul_distance(address1, address2):
    geolocator = Nominatim(user_agent="distance_calculator")
    location1 = geolocator.geocode(address1)
    location2 = geolocator.geocode(address2)

    if not location1 or not location2:
        raise ValueError("One or both addresses could not be geocoded.")

    coords_1 = (location1.latitude, location1.longitude)
    coords_2 = (location2.latitude, location2.longitude)

    distance_kilometers = geodesic(coords_1, coords_2).kilometers
    return distance_kilometers


def nearest_Doctors(patient_address, Specialty):
    dataset = pd.read_excel('./dataset.xlsx') 
    df = dataset[dataset['Speciality'] == Specialty]
    list_of_ids = df['Med_ID'].tolist()
    
    distances = []
    for ID in list_of_ids: 
        row = dataset[dataset['Med_ID'] == ID]
        Med_address = row['Med_Address'].iloc[0]
        distance = calcul_distance(Med_address, patient_address)
        distances.append(distance) 

    tableau = np.array([list_of_ids, distances])
    sorted_indices = np.argsort(tableau[1])
    sorted_tableau = tableau[:, sorted_indices]
    ids_nearest_doc = sorted_tableau[0, :4]
    
    coordonnee_med = df[df['Med_ID'].isin(ids_nearest_doc)]
    return coordonnee_med

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
        print("DoCare (Meditron): ", meditron_response)
        
        context += f"\nUser: {user_input}\nDocare (Meditron): {meditron_response}\nDocare (Modified): {chatbot_response}\n"
        
        if "address" in user_input.lower():
            print("Could you please provide your address so that I can recommend the nearest doctors you can consult?")
            patient_address = input("You: ")
            Speciality = "Cardiologie"  # Update this based on user input if necessary
            nearest_docs = nearest_Doctors(patient_address, Speciality)
            print(nearest_docs)

if __name__ == "__main__":
    handleConversation()
