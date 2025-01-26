import requests

def get_evaluation():
    response = requests.get('http://127.0.0.1:8080/evaluate_qa')
    return response.json()

if __name__ == '__main__':
    results = get_evaluation()
    print(results)