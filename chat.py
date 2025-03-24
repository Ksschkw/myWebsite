import random
import json
import torch
import requests
import os
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words, context_vector, prepare_full_input
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_TOKEN = os.getenv("API_KEY")
bot_name = "KOSISOCHUKWUBOT"

def get_fallback_response(user_input):
    models = [
        ('deepseek/deepseek-r1:free', 8),
        ('mistralai/mistral-7b-instruct-v0.2', 10),
        ('openai/gpt-3.5-turbo', 12)
    ]
    
    for model_name, timeout in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_TOKEN}"},
                json={
                    "model": model_name,
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
                },
                timeout=timeout
            )
            return response.json()['choices'][0]['message']['content']
        except:
            continue
            
    return "Let's talk about my projects instead! What would you like to know?"

def initialize_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    
    model_data = torch.load('best_model.pth', map_location=device)
    model = NeuralNet(
        input_size=model_data['input_size'],
        hidden_size=model_data['hidden_size'],
        num_classes=model_data['output_size']
    ).to(device)
    model.load_state_dict(model_data['model_state'])
    model.eval()
    
    return {
        'model': model,
        'intents': intents,
        'model_data': model_data,
        'device': device
    }

def process_message(user_input, components):
    tokens = tokenize(user_input)
    bow = bag_of_words(tokens, components['model_data']['all_words'])
    bow = torch.from_numpy(bow).float().unsqueeze(0).to(components['device'])
    
    with torch.no_grad():
        output = components['model'](bow)
    
    prob = torch.softmax(output, dim=1)
    top_prob, top_idx = torch.max(prob, dim=1)
    
    if top_prob.item() > 0.8918:
        for intent in components['intents']['intents']:
            if intent['tag'] == components['model_data']['tags'][top_idx.item()]:
                return random.choice(intent['responses'])
    
    return get_fallback_response(user_input)