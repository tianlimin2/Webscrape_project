from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import datetime

class QAEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_log = []
        
    def log_qa(self, question, answer, search_results):
        timestamp = datetime.datetime.now().isoformat()
        qa_pair = {
            'timestamp': timestamp,
            'question': question,
            'answer': answer,
            'search_results': search_results,
            'metrics': self.calculate_metrics(question, answer, search_results)
        }
        self.qa_log.append(qa_pair)
        self.save_logs()
        
    def calculate_metrics(self, question, answer, search_results):
        # Question-Answer relevance
        qa_similarity = float(util.cos_sim(
            self.model.encode(question),
            self.model.encode(answer)
        ))
        
        # Answer-Context relevance
        context = ' '.join(search_results)
        ac_similarity = float(util.cos_sim(
            self.model.encode(answer),
            self.model.encode(context)
        ))
        
        # Answer quality metrics
        answer_length = len(answer.split())
        has_structure = bool(any(answer.startswith(str(i)) for i in range(1, 10)))
        
        return {
            'qa_similarity': qa_similarity,
            'context_relevance': ac_similarity,
            'answer_length': answer_length,
            'has_structure': has_structure
        }
    
    def save_logs(self):
        df = pd.DataFrame(self.qa_log)
        df.to_csv('qa_evaluation.csv', index=False)

# 在 server.py 中集成
evaluator = QAEvaluator()

# 修改 search 函数:
def search():
    request_data = request.get_json()
    search_query = request_data['search']
    
    top_results = semantic_search(search_query, data)
    combined_results = "\n".join(top_results)
    
    ai_response = ask_azure_openai(deployment_name, prompt)
    
    # Log and evaluate
    evaluator.log_qa(search_query, ai_response, top_results)
    
    return {"code": 200, "data": {"search": search_query, "answer": ai_response}}