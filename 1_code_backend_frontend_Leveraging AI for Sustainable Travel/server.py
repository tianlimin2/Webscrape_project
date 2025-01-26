from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from openai import AzureOpenAI
import torch
import json
import os
from evaluate import QAEvaluator

# Initialize Flask and environment
app = Flask(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configuration
CSV_FILE = 'processed_articles.csv'
EMBEDDINGS_FILE = 'embeddings.pt' 
QA_FILE = 'qa.csv'

# Initialize models and load data
model = SentenceTransformer('all-MiniLM-L6-v2')
data = pd.read_csv(CSV_FILE)
data.columns = data.columns.str.lower()
qa_data = pd.read_csv(QA_FILE)
evaluator = QAEvaluator()

# Load embeddings
if os.path.exists(EMBEDDINGS_FILE):
   embeddings = torch.load(EMBEDDINGS_FILE, weights_only=True, map_location='cpu')
   if len(embeddings) == len(data):
       data['embedding'] = [tensor.cpu().numpy() for tensor in embeddings]
   else:
       data['embedding'] = data['content'].apply(lambda x: model.encode(x, convert_to_tensor=True).cpu().numpy())
       torch.save([torch.from_numpy(emb) for emb in data['embedding']], EMBEDDINGS_FILE)
else:
   if 'content' in data.columns:
       data['embedding'] = data['content'].apply(lambda x: model.encode(x, convert_to_tensor=True).cpu().numpy())
       torch.save([torch.from_numpy(emb) for emb in data['embedding']], EMBEDDINGS_FILE)
   else:
       raise ValueError("CSV must contain a 'content' column")

# Initialize Azure OpenAI
with open('key.txt', 'r') as f:
   config = json.load(f)

client = AzureOpenAI(
   api_key=config['api_key'],
   api_version=config['api_version'],
   azure_endpoint=config['api_base']
)
deployment_name = config['deployment_name']

def semantic_search(query, data, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    similarities = [float(util.cos_sim(query_embedding, embedding)) for embedding in data['embedding']]
    data['similarity'] = similarities
    top_matches = data.nlargest(top_k, 'similarity')
    return top_matches['content'].tolist()
def ask_azure_openai(deployment_name, question):
   response = client.chat.completions.create(
       model=deployment_name,
       messages=[
           {"role": "system", "content": "You are a helpful assistant focused on sustainable travel."},
           {"role": "user", "content": question}
       ]
   )
   return response.choices[0].message.content

# Routes
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/evaluate_qa', methods=['GET'])
def evaluate_qa():
   results = []
   for _, row in qa_data.iterrows():
       question = row['question']
       top_results = semantic_search(question, data)
       combined_results = "\n".join(top_results)
       
       prompt = f"Based on the following information about eco-travel, answer the question: '{question}'\n\nInformation:\n{combined_results}"
       ai_response = ask_azure_openai(deployment_name, prompt)
       
       evaluator.log_qa(question, ai_response, top_results)
       results.append({
           'question': question,
           'answer': ai_response
       })
   
   return {'code': 200, 'data': results}

@app.route('/search', methods=['POST'])
def search():
   request_data = request.get_json()
   search_query = request_data['search']
   
   top_results = semantic_search(search_query, data)
   combined_results = "\n".join(top_results)
   
   prompt = f"Based on the following information about eco-travel, answer the question: '{search_query}'\n\nInformation:\n{combined_results}"
   ai_response = ask_azure_openai(deployment_name, prompt)
   
   evaluator.log_qa(search_query, ai_response, top_results)
   
   return {"code": 200, "data": {"search": search_query, "answer": ai_response}}

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)