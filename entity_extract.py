import logging
import json
from typing import Any, Dict, List, Protocol
from abc import ABC, abstractmethod
import os
from openai import OpenAI
import spacy
from dateutil.parser import parse as parse_date, ParserError
import aisuite as ai
# Optional: Use OpenAI or HuggingFace Transformers if available
try:
    import openai
    from transformers import pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Strategy Pattern for Entity Extraction ----

class EntityExtractionStrategy(ABC):
    @abstractmethod
    def extract(self, text: str) -> Dict[str, List[str]]:
        pass


class SpaCyEntityExtractor(EntityExtractionStrategy):
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            logger.debug(f"Loading spaCy model '{model}'")
            self.nlp = spacy.load(model)
            logger.info(f"spaCy model '{model}' loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load spaCy model '{model}': {e}")
            raise

    def extract(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        results = {
            "Person": [],
            "Date": [],
            "Location": [],
            "Organization": [],
            "Event": []
        }

        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            if label == "PERSON":
                results["Person"].append(value)
            elif label in {"DATE", "TIME"}:
                try:
                    parsed = parse_date(value, fuzzy=True)
                    normalized = parsed.date().isoformat()
                    results["Date"].append(normalized)
                except (ParserError, ValueError):
                    results["Date"].append(value)
            elif label in {"GPE", "LOC"}:
                results["Location"].append(value)
            elif label == "ORG":
                results["Organization"].append(value)
            elif label in {"EVENT", "WORK_OF_ART"}:
                results["Event"].append(value)

        return {k: list(dict.fromkeys(v)) for k, v in results.items()}


class OpenAIEntityExtractor(EntityExtractionStrategy):
    def __init__(self, model: str = "openai:gpt-4o-mini"):
    #    "gpt-4o-mini"
        self.client = ai.Client()
        self.model = model

    def extract(self, text: str) -> Dict[str, List[str]]:
        prompt = f"""
        Extract the following structured information from this text and return JSON:
        Fields:
        - Person
        - Date
        - Location
        - Organization
        - Event

        Text:
        {text}

        Respond only with JSON.
        """
        try:
            # "groq:llama-3.1-8b-instant" 
            # self.client = ai.Client()
            # response = self.client.chat.completions.create(model=self.model, messages=prompt)
            # response_content = response.choices[0].message.content
            # print(response_content)
            self.client = OpenAI()
            response = self.client.responses.create(
            model=self.model,
            instructions="You are an information extractor.",
            input=prompt,
            )

            print(response.output_text)
             
        except Exception as e:
            logger.exception("OpenAI extraction failed")
            return None


class BERTEntityExtractor(EntityExtractionStrategy):
    def __init__(self):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers not available")
        self.ner = pipeline("ner", grouped_entities=True)

    def extract(self, text: str) -> Dict[str, List[str]]:
        entities = self.ner(text)
        results = {
            "Person": [],
            "Date": [],
            "Location": [],
            "Organization": [],
            "Event": []
        }

        for ent in entities:
            label = ent['entity_group']
            value = ent['word']
            if "PER" in label:
                results["Person"].append(value)
            elif "ORG" in label:
                results["Organization"].append(value)
            elif "LOC" in label:
                results["Location"].append(value)
            elif "MISC" in label:
                results["Event"].append(value)

        return {k: list(dict.fromkeys(v)) for k, v in results.items()}


# ---- Context / Facade ----
class EntityExtractionContext:
    def __init__(self, strategy: EntityExtractionStrategy):
        self.strategy = strategy

    def extract(self, text: str) -> Dict[str, List[str]]:
        return self.strategy.extract(text)


# ---- CLI ----
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured data from text.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input text file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--engine", type=str, default="spacy", choices=["spacy", "openai", "bert"], help="Engine to use.")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    try:
        if args.engine == "spacy":
            extractor = SpaCyEntityExtractor()
        elif args.engine == "openai":
            extractor = OpenAIEntityExtractor()
        elif args.engine == "bert":
            extractor = BERTEntityExtractor()
        else:
            raise ValueError("Unsupported engine")

        context = EntityExtractionContext(extractor)
        extracted = context.extract(text)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(extracted, f, indent=2)
        logger.info(f"Extracted entities written to {args.output}")

    except Exception as e:
        logger.exception("Extraction failed")
        exit(1)


if __name__ == "__main__":
    main()
