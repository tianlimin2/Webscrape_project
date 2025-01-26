# Leveraging AI for Sustainable Travel

This project aims to leverage artificial intelligence technologies to promote sustainable travel. It includes a Flask-based backend service that uses semantic search and Azure OpenAI to answer questions about eco-friendly travel.

## Features

- Semantic Search: Utilizes SentenceTransformer model for similarity matching
- Q&A System: Generates answers about sustainable travel using Azure OpenAI
- Evaluation Function: Assesses the performance of the Q&A system
- RESTful API: Provides API endpoints for search and Q&A functionality

## Project Structure

- `server.py`: Main Flask application file
- `processed_articles.csv`: Contains processed article data
- `embeddings.pt`: Pre-computed article embedding vectors
- `qa.csv`: Question-answer data for evaluation
- `key.txt`: JSON file containing Azure OpenAI configuration
- `evaluate.py`: Evaluation module containing the QAEvaluator class

## Installation

1. Clone the repository:
git clone

2. Install dependencies:
pip install -r requirements.txt

3. Configure Azure OpenAI:
Add your Azure OpenAI configuration to the `key.txt` file.

## Usage

1. Run the Flask application:
python server.py

2. Access the API:
- Home: `GET /`
- Evaluate Q&A: `GET /evaluate_qa`
- Search: `POST /search`

## Configuration

- `CSV_FILE`: Path to the processed articles data file
- `EMBEDDINGS_FILE`: Path to the embedding vectors file
- `QA_FILE`: Path to the Q&A evaluation data file

## Dependencies

- Flask
- sentence-transformers
- pandas
- openai
- torch
- For other dependencies, please refer to `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

For any questions or suggestions, please contact the project maintainer.
