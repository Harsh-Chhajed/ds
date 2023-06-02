import openai
API_KEY = 'sk-'
openai.api_key = API_KEY
model = "text-davinci-003"  # The GPT model to use

def generate_response(prompt):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=250,  # Adjust the response length as needed
        temperature=0.7,  # Controls the randomness of the response. Higher values = more random
        n=1,  # Number of responses to generate
    )
    return response.choices[0].text.strip()

# Main chat loop
print('Chatbot is ready. Start chatting!')
while True:
    user_input = input('User: ')
    prompt = f'User: {user_input}\nChatbot: '
    response = generate_response(prompt)
    print('Chatbot:', response)
