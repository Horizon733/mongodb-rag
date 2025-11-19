# MongoDB RAG Streamlit App

## Prerequisites

- Python 3.8+
- MongoDB running locally (`mongodb://localhost:27017`)
- Ollama running locally (`http://localhost:11434`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start MongoDB**  
   Make sure MongoDB Atlas is running locally.
   ```bash
   docker run -p 27017:27017 mongodb/mongodb-atlas-local
   ```

2. **Start Ollama**  
   Make sure Ollama is running and the model (`llama3.1`) is available. Feel free to use your choice of LLM model
   ```bash
   ollama pull llama3.1
   ```

3. **Run the Streamlit app**  
   In your terminal:

   ```bash
   streamlit run main.py
   ```

4. **Upload a PDF**  
   Use the sidebar to upload a PDF. The app will chunk and embed the text into MongoDB.

5. **Ask Questions**  
   Enter your question in the main input box. The app will retrieve relevant context and generate an answer using Ollama.

## Notes

- The first run will create a vector search index in MongoDB if it does not exist.
- Ensure your MongoDB version supports vector search and `SearchIndexModel`.
- The embedding model used is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
