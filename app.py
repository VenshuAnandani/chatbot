import joblib
import random
import json
import streamlit as st
import os
from datetime import datetime

# Load the saved vectorizer and classifier
vectorizer = joblib.load("vectorizer.joblib")
clf = joblib.load("clf.joblib")

# Load the intents JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)["intents"]

# Define the chatbot function with a new name to avoid conflict
def get_chatbot_response(input_text):
    input_text = vectorizer.transform([input_text])  # Transform input text to vector
    tag = clf.predict(input_text)[0]  # Predict the tag
    for intent in intents:
        if intent['intent'] == tag:  # Match the predicted tag to the intent
            response = random.choice(intent['responses'])  # Random response
            return response
    return "I'm sorry, I didn't understand that."

# Function to save chat history to a file with timestamp
def save_chat_history(chat_history):
    if not os.path.exists("chat_logs"):
        os.makedirs("chat_logs")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"chat_logs/chat_{timestamp}.json", "w") as file:
        json.dump(chat_history, file)

# Function to load chat histories
def load_chat_histories():
    chat_histories = []
    if os.path.exists("chat_logs"):
        for file_name in os.listdir("chat_logs"):
            if file_name.endswith(".json"):
                try:
                    with open(os.path.join("chat_logs", file_name), "r") as file:
                        # Skip empty or invalid files
                        if os.stat(os.path.join("chat_logs", file_name)).st_size == 0:
                            continue
                        chat_histories.append({
                            "timestamp": file_name.split("_")[1].replace(".json", ""),
                            "history": json.load(file),
                        })
                except json.JSONDecodeError:
                    # Skip files with invalid JSON
                    continue
    return chat_histories


# Streamlit app
def main():
    # Set page config for dark mode and title
    st.set_page_config(page_title="Void Main Pvt. Ltd. | Support Chatbot", layout="centered")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # App title and welcome message
    st.title("Void Main Pvt. Ltd.")
    st.write("Welcome to the support chatbot, how can I help you?")

    # User input section
    user_input = st.text_input("You:", "", key="user_input")

    # Display chatbot response and maintain history
    if user_input:
        response = get_chatbot_response(user_input)  # Get chatbot response
        # Add message with timestamp
        st.session_state["chat_history"].append({
            "user": user_input,
            "bot": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Display chat history
    for chat_item in st.session_state["chat_history"]:
        # Validate that all expected keys are present
        if all(key in chat_item for key in ["timestamp", "user", "bot"]):
            st.write(f"**You:** {chat_item['user']}")
            st.write(f"**Chatbot:** {chat_item['bot']}")
        else:
            st.write("Error displaying chat item: Missing required fields.")

    # Option to view past chat histories
    if st.button("View Past Chats"):
        past_chats = load_chat_histories()
        if past_chats:
            for chat in past_chats:
                for message in chat['history']:
                    st.write(f"**[{message['timestamp']}]**")
                    # Validate the message structure
                    if all(key in message for key in ["timestamp", "user", "bot"]):
                        st.write(f"**You:** {message['user']}")
                        st.write(f"**Chatbot:** {message['bot']}")
                    else:
                        st.write("Error displaying past message: Missing required fields.")
        else:
            st.write("No past chats found.")

    # Add a save chat button
    if st.button("Save Chat"):
        if st.session_state["chat_history"]:
            save_chat_history(st.session_state["chat_history"])
            st.write("Chat history saved successfully!")
        else:
            st.write("No chat history to save.")

if __name__ == "__main__":
    main()
