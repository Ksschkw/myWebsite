# chat.py
import random
import json
import torch
from collections import deque
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words, context_vector, prepare_full_input
# Add these imports at the top
import requests
import random
import os
from dotenv import load_dotenv


load_dotenv()
OPENROUTER_TOKEN = os.getenv("API_KEY")

def get_fallback_response(user_input):
    """Try multiple free LLM APIs in sequence"""
    try:
        print('going to first fallback')
        print('Trying deepseek-R1 on open router')
        response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
                json={
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"""Hello i'm {bot_name}, and that will be my name throughout this conversation. I will always respond in a brief manner, as brief as possible. Also this is a little bit about me, that i will always consider first before answering any of your prompts:
                            Hello! I'm Kosisochukwu, a passionate and driven individual with a knack for all things technical and creative. Currently, I'm a software engineering student in my fourth year at the Federal University of Technology Owerri, where I'm honing my skills and preparing to make a significant impact in the tech world.

                            When I'm not deep into my studies or writing Python code, you can find me indulging in puzzles like Sudoku and jigsaws, or relaxing with some anime. These hobbies keep me sharp, focused, and inspired.

                            One of my core values is the relentless pursuit of excellence—I simply refuse to be average. This mindset pushes me to strive for greatness in everything I do, whether it's acing a project, solving a complex puzzle, or building innovative software solutions.

                            What gets me out of bed every morning? The thrill of creation and the quest for financial independence. There's nothing quite like the sense of fulfillment I experience when I build something from scratch and watch it work as intended. It's a feeling that drives me to keep pushing boundaries and seeking new challenges.

                            I'm constantly evolving, learning, and aiming to be the best version of myself. Welcome to my journey.
                            Meet Kosisochukwu (aka Kosi), a 400-level Software Engineering student at the Federal University of Technology Owerri, blending academic rigor with tech prowess. Proficient in Python, JavaScript, HTML/CSS, and dabbling in Java/C++, Kosi crafts basic websites, custom chatbots, and machine learning projects (NLP, CNNs, RNNs) with a flair for creativity. A self-proclaimed Naruto superfan 🍥, he codes side projects, binge anime, and draw inspiration from the Hokage’s persistence and the grind for financial independence. Open for collaborations and opportunities, Kosi invites you to connect via email, WhatsApp, or GitHub (@Ksschkw). Check out their portfolio at kosisochukwu.onrender.com and drop a "Dattebayo!" on X (@_Kosisochuk) for a mix of tech talk and anime vibes. 🚀 "Living my best digital life!"

                            Casual, tech-savvy, and always open to building something cool—no Rasengan required. 😎
                            """
                        },
                        {
                        "role": "user", 
                        "content": f"""You're {bot_name}'s assistant. Respond briefly to: {user_input} and do not ever use {bot_name} in your replies, EVER. and remember to be brief. and remember to make sure you're not using {bot_name} in your replies and do not mention that you are omitting {bot_name} ever, you should never speak of this. And remember to use this about:
                        Meet Kosisochukwu (aka Kosi), a 400-level Software Engineering student at the Federal University of Technology Owerri, blending academic rigor with tech prowess. Proficient in Python, JavaScript, HTML/CSS, and dabbling in Java/C++, Kosi crafts basic websites, custom chatbots, and machine learning projects (NLP, CNNs, RNNs) with a flair for creativity. A self-proclaimed Naruto superfan 🍥, he codes side projects, binge anime, and draw inspiration from the Hokage’s persistence and the grind for financial independence. Open for collaborations and opportunities, Kosi invites you to connect via email, WhatsApp, or GitHub (@Ksschkw). Check out their portfolio at kosisochukwu.onrender.com and drop a "Dattebayo!" on X (@_Kosisochuk) for a mix of tech talk and anime vibes. 🚀 "Living my best digital life!"

                        Casual, tech-savvy, and always open to building something cool—no Rasengan required. 😎
                        """
                    }]
                }
            )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        try:
            # Fallback to OpenRouter
            print('going to second fallback..mistralai/mistral-7b-instruct-v0.2 on openrouter')
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
                json={
                    "model": "mistralai/mistral-7b-instruct-v0.2",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"Hello i'm {bot_name}"
                        },
                        {
                        "role": "user", 
                        "content": f"You're {bot_name}'s assistant. Respond briefly to: {user_input} and do not ever use {bot_name} in your replies, EVER."
                    }]
                }
            )
            return response.json()['choices'][0]['message']['content']
        except:
            try:
                print('going to final fallback.. gpt3 on open router')
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
                    json={
                        "model": "mistralai/mistral-7b-instruct-v0.2",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"Hello i'm {bot_name} and that will be my name throughout this conversation. I will always respond in a brief manner, as brief as possible."
                            },
                            {
                            "role": "user", 
                            "content": f"You're {bot_name}'s assistant. Respond briefly to: {user_input} and do not ever use {bot_name} in your replies, EVER."
                        }]
                    }
                )
                return response.json()['choices'][0]['message']['content']
            except:
                return "I'm still learning! Ask me about my projects instead."

class ChatContextManager:
    def __init__(self, max_history=3):
        self.active_contexts = set()
        self.context_history = deque(maxlen=max_history)
        self.pending_requirements = set()  # Stores tuples

    def update_context(self, new_contexts):
        self.context_history.append(self.active_contexts.copy())
        self.active_contexts = set(new_contexts)
        self.pending_requirements.clear()

    def add_requirement(self, requirements):
        if requirements:
            # Convert list to tuple for hashability
            req_tuple = tuple(requirements) if isinstance(requirements, list) else requirements
            self.pending_requirements.add(req_tuple)

    def validate_requirements(self):
        return all(
            req in self.active_contexts
            for req_tuple in self.pending_requirements
            for req in req_tuple
        )

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and data
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

model_data = torch.load('best_model.pth', map_location=device)
model = NeuralNet(
    input_size=model_data['input_size'],
    hidden_size=model_data['hidden_size'],
    num_classes=model_data['output_size'],
    context_size=len(model_data.get('context_tags', []))
).to(device)
model.load_state_dict(model_data['model_state'])
model.eval()

# Initialize context system
context_manager = ChatContextManager()
all_context_tags = model_data.get('context_tags', [])
BOW_SIZE = len(model_data['all_words'])

bot_name = "KOSISOCHUKWUBOT"
print(f"{bot_name}: Ready for interaction! Type 'reset' to clear context or 'quit' to exit.")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'reset':
        context_manager = ChatContextManager()
        print(f"{bot_name}: Context reset complete")
        continue

    # Preprocess input
    tokens = tokenize(user_input)
    bow = bag_of_words(tokens, model_data['all_words'])
    ctx_vec = context_vector(context_manager.active_contexts, all_context_tags)
    full_input = torch.from_numpy(prepare_full_input(bow, ctx_vec)).float().unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(full_input)
    
    prob = torch.softmax(output, dim=1)
    top_prob, top_idx = torch.max(prob, dim=1)
    predicted_tag = model_data['tags'][top_idx.item()]

    if top_prob.item() > 0.8918:
        response_given = False
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                # Select response
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                response_given = True
                break
        
        if not response_given:
            print(f"{bot_name}: I'm not sure how to respond to that in the current context")
    else:
        fallback_response = get_fallback_response(user_input)
        print(f"{bot_name}: {fallback_response}")

    # Auto-context cleanup after 5 inactive turns
    if len(context_manager.context_history) >= 5 and not context_manager.active_contexts:
        context_manager.update_context([])
        print(f"{bot_name}: [System] Conversation context automatically refreshed")