📦 Entity Extractor
Extract structured information such as People, Dates, Locations, Organizations, and Events from unstructured text using multiple AI models (spaCy, OpenAI GPT, Hugging Face BERT).

🚀 Features
✅ Supports modular AI backends (pluggable via CLI):

spaCy (local NLP)

OpenAI GPT (LLM with prompt chaining)

BERT from HuggingFace (NER pipeline)

✅ Strategy design pattern for extensibility

✅ Outputs normalized, structured JSON

✅ CLI-friendly — choose input, output, and engine

✅ Logging & graceful error handling

📁 Project Structure
graphql
Copy
Edit
├── entity_extractor.py   # Main extraction script (modular and extensible)
├── input.txt             # Sample input text
├── output_spacy.json     # Sample structured output
├── README.md             # You're here!
⚙️ Setup
1. 🔧 Install dependencies
bash
Copy
Edit
pip install spacy openai transformers python-dateutil
python -m spacy download en_core_web_sm
2. 🔑 If using OpenAI, set your API key
bash
Copy
Edit
export OPENAI_API_KEY=your_openai_key
🧠 Usage
💬 Sample input: input.txt
txt
Copy
Edit
Steve Jobs founded Apple in 1976 in California. Tim Cook now leads it. Major events like WWDC happen in San Jose.
🛠️ Run the extractor
➤ Using spaCy:
bash
Copy
Edit
python entity_extractor.py -i input.txt -o output_spacy.json --engine spacy
➤ Using OpenAI GPT:
bash
Copy
Edit
python entity_extractor.py -i input.txt -o output_openai.json --engine openai
➤ Using BERT (Hugging Face):
bash
Copy
Edit
python entity_extractor.py -i input.txt -o output_bert.json --engine bert
📤 Output: output_*.json
json
Copy
Edit
{
  "Person": ["Steve Jobs", "Tim Cook"],
  "Date": ["1976"],
  "Location": ["California", "San Jose"],
  "Organization": ["Apple"],
  "Event": ["WWDC"]
}
🏗️ How It Works
Uses the Strategy pattern to abstract model implementations

Each extractor class implements .extract(text) and returns a unified JSON schema

CLI lets you switch strategies without code changes

OpenAIEntityExtractor uses prompt chaining and instructs the LLM to return structured JSON

BERTEntityExtractor uses HuggingFace's pre-trained models

🔌 Extending the Tool
Add a new model or approach:

Subclass EntityExtractionStrategy

Implement extract(self, text: str)

Plug it into the CLI via the --engine flag

📚 Future Ideas
 Add PDF/HTML input parsing

 Deploy as FastAPI microservice

 Add schema validation (e.g., with Pydantic)

 Integrate LangChain for prompt chaining & memory

