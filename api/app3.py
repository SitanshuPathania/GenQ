from flask import Flask, render_template, request, redirect, flash, session
import pymongo
import torch
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import gdown
import os

app = Flask(__name__)
app.secret_key = '5234'

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://sitanshupathania5234:DBProjects@cluster0.fveeici.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["GenQ"]
LoginInformation = db["LoginInformation"]
RegistrationInfo = db["RegistrationInfo"]
QuizInfo = db["QuizInfo"]
FeedbackInfo = db["FeedbackInfo"]

# Download model from Google Drive
def download_model_from_drive(file_id, destination):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Google Drive file IDs
QG_MODEL_CHECKPOINT_ID = '1sUXlj6yIDOc-1C3EC2GiRTb3sAtVoCZb'  # QG model
DSTR_MODEL_CHECKPOINT_ID = '1ev2rxADmQRcbfmWwo0lYkT9jCYSJSFEp'  # DSTR model

# Model paths
QG_MODEL_CHECKPOINT_PATH = './models/best-checkpoint-v4.ckpt'
DSTR_MODEL_CHECKPOINT_PATH = './models/best-checkpoint-v16.ckpt'

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Download models if they don't exist
if not os.path.exists(QG_MODEL_CHECKPOINT_PATH):
    download_model_from_drive(QG_MODEL_CHECKPOINT_ID, QG_MODEL_CHECKPOINT_PATH)
if not os.path.exists(DSTR_MODEL_CHECKPOINT_PATH):
    download_model_from_drive(DSTR_MODEL_CHECKPOINT_ID, DSTR_MODEL_CHECKPOINT_PATH)

# Load models
QGmodel = T5ForConditionalGeneration.from_pretrained(QG_MODEL_CHECKPOINT_PATH)
DstrModel = T5ForConditionalGeneration.from_pretrained(DSTR_MODEL_CHECKPOINT_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Homepage.html')
def homepage():
    return render_template('Homepage.html')

@app.route('/process_form', methods=['GET', 'POST'])
def process_form():
    if request.method == 'POST':
        # Limit the input size
        para = request.form['para'][:512]  # Limit comprehension to 512 characters
        ans = request.form['ans'][:100]  # Limit answer to 100 characters
        
        if len(para) > 512 or len(ans) > 100:
            flash('Input too long, please shorten your text.', 'error')
            return redirect('/Homepage.html')

        quiz = generate(ans, para)
        QuizInfo.insert_one({'comprehension': para, 'answer': ans})
        return render_template('quiz.html', quiz=quiz)
    else:
        return "Method not allowed", 405

def generate(answer, context):
    ques = generateQuestion(QGmodel, answer, context)
    
    # Limit the output size
    flag = 0
    question = []
    for word in ques.split():
        if '</s>' in word:
            word = word.replace('</s>', '')
        if flag == 1:
            question.append(word)
        if '<sep>' in word:
            flag = 1
    question = ' '.join(question)[:200]  # Limit question to 200 characters

    optns = generateOptions(DstrModel, answer, question, context)
    
    # Extracting options and limiting their size
    flag = 0
    options = []
    temp = []
    optns = optns.replace('<', ' <')
    for word in optns.split():
        if word == '<pad>':
            continue
        if word == '<sep>':
            if temp:
                options.append(' '.join(temp)[:100])  # Limit each option to 100 characters
            temp = []
            continue
        if '<extra_id' in word:
            if temp:
                options.append(' '.join(temp)[:100])  # Limit each option to 100 characters
            temp = []
            continue
        temp.append(word)
    return ques, options

def generateQuestion(model, answer, context):
    # Your implementation for generating questions
    # This is a placeholder; replace with actual code
    input_text = f"{context} <extra_id_0> {answer}"
    input_ids = T5Tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids)
    return T5Tokenizer.decode(output[0], skip_special_tokens=True)

def generateOptions(model, answer, question, context):
    # Your implementation for generating options
    # This is a placeholder; replace with actual code
    input_text = f"{context} <extra_id_0> {answer} <extra_id_1> {question}"
    input_ids = T5Tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids)
    return T5Tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
