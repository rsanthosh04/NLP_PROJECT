import os
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the BART model for summarization and GPT-Neo for text generation
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")  # GPT-Neo model

# Function to summarize text
def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to generate text based on custom words or phrases
def generate_text(prompt):
    # Provide clear instructions in the prompt for better relevance
    adjusted_prompt = f"Generate a detailed text based on the following idea: {prompt}"
    generated = text_generator(adjusted_prompt, max_length=150, num_return_sequences=1, temperature=0.7, top_k=50)
    return generated[0]['generated_text'].strip()

# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    user_input = request.json['text']
    summary = summarize_text(user_input)
    return jsonify({'summary': summary})

# Flask route for text generation with custom words
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['text']
    generated_text = generate_text(prompt)
    return jsonify({'generated_text': generated_text})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
