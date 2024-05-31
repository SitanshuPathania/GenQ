import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
# import pymongo
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from flask import Flask, render_template, request

app = Flask(__name__)

# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["GenQ"]
# collection = db["LoginInfo"]

pl.seed_everything(42)
MODEL_NAME = 't5-small'
LEARNING_RATE = 0.0001
SEP_TOKEN = '<sep>'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(SEP_TOKEN)
TOKENIZER_LEN = len(tokenizer)

class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN) #resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
  
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)

checkpoint_path = "./models/best-checkpoint-v4.ckpt"
QGmodel = QGModel.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
QGmodel.freeze()
checkpoint_path = "./models/best-checkpoint-v16.ckpt"
DstrModel = QGModel.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
DstrModel.freeze()

def generateQuestion(qgmodel: QGModel, answer: str, context: str) -> str:
    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=300,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)

def generateOptions(qgmodel: QGModel, answer: str, question: str, context: str) -> str:
    source_encoding = tokenizer(
        '{} {} {} {} {}'.format(answer, SEP_TOKEN, question, SEP_TOKEN, context),
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=64,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)

def generate(answer, context):
    ques = generateQuestion(QGmodel, answer, context)
    # Extracting question
    flag = 0
    question = []
    for word in ques.split():
        if '</s>' in word:
            word = word.replace('</s>', '')
        if flag == 1:
            question.append(word)
        if '<sep>' in word:
            flag = 1
    question = ' '.join(question)
    optns = generateOptions(DstrModel, answer, question, context)
    # Extracting options
    flag = 0
    options = []
    temp = []
    optns = optns.replace('<', ' <')
    for word in optns.split():
        if word == '<pad>':
            continue
        if word == '<sep>':
            options.append(' '.join(temp))
            temp = []
            continue
        if '<extra_id' in word:
            options.append(' '.join(temp))
            temp = []
            continue
        if word == '</s>':
            options.append(' '.join(temp))
            temp = []
            continue
        temp.append(word)
    # options = {'correct': answer, 'incorrect': options}
    final_text = question + '\n'
    opt = {'a': answer}
    i = 1
    for o in options:
        opt[chr(97+i)] = o
        i = i+1
    final_text = 'Question: ' + question
    for keys in opt.keys():
        final_text = final_text + '\n' + keys + '. ' + opt[keys]
    final_text = final_text + '\n\nCorrect Options is a.'
    return final_text

@app.route("/", methods=['GET', 'POST'])
def fun():
    if request.method=='POST':
        data = request.form
        para = data['para']
        ans = data['ans']
        quiz = generate(ans, para)
        return render_template('quiz.html', quiz = quiz)
    return render_template('Index2.html')

@app.route('/quiz')
def fun2():
    return render_template('quiz.html')

if __name__=="__main__":
    app.run(debug=True)