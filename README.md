# RAG (Retrieval Augmented Generation) Playground

Playing with different tools for building RAG systems.
- `Embeddings`: OpenAI, Huggingface
- `Vector Stores`: ChromaDB, QDrant
- `LLM`: OpenAI, Huggingface
- `Chat interfaces`: Gradio, Streamlit, Typer
- `LLM Frameworks`: Langchain, Llamaindex
- `Prompting languages`: LMQL, Guidance
- `Agents`

## Langchain + Huggingface or OpenAI

### Index data
- Put the documents you want to index inside doc folder.
- Run `create_index.py`

### Start and mlflow server
Start the server for working locally:
```bash
mlflow server --backend-store-uri <absolute-path-to-tracking-directory> --default-artifact-root tracking/ --host 127.0.0.1:5001
```
Open the UI by going to `http://127.0.0.1:5001/`

### Ask questions
- Change the query.
- Run `chatbot.py`

## TODO
- Include MLflow for tracking RAG experiments with different configurations.
    - https://dagshub.com/blog/mlflow-support-for-large-language-models/
    - https://mlflow.org/docs/latest/python_api/mlflow.llm.html#mlflow.llm.log_predictions (prompts, inputs (questions, context) )
    - https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_text (documents)
    - https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param log the number of tokens per answer, number of input tokens sent to bot too (in case of using bedrock who charges by the number input and output tokens)
    - https://medium.com/@dliden/evaluating-retrieval-augmented-generation-rag-systems-with-mlflow-cf09a74faadb
    - https://mlflow.org/docs/latest/models.html#evaluating-with-llms
- Include ragas to evaluate queantitatively performance of both embeddings and LLMs in the RAG.
    - https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
- Move hardcoded values in both files to a central configurations file where this is defined.
- Include Bedrock models for both embeddings and LLMs with langchain.
    - https://aws.amazon.com/blogs/aws/preview-connect-foundation-models-to-your-company-data-sources-with-agents-for-amazon-bedrock/
    - https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/03_QuestionAnswering/01_qa_w_rag_claude.ipynb
    - https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/04_Chatbot/00_Chatbot_Claude.ipynb
- Include a typer interface to chat from the terminal
- Include streamlit or gradio interface to interactively:
    - Load new documents and create to different collections and with different metadata key-value pairs.
    - Chat with the documents, keep the history and choose the collection.
