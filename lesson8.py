import warnings
import numpy as np
import re
from typing import Optional
# from IPython.display import clear_output

from openai import OpenAI
from guardrails import Guard
from guardrails.validator_base import (
                                        FailResult,
                                        PassResult,
                                        ValidationResult,
                                        Validator,
                                        register_validator,
                                        )
from guardrails.errors import ValidationError
from transformers import pipeline

from helper import RAGChatWidget, SimpleVectorDB
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")
%env TOKENIZERS_PARALLELISM = true

## ------------------------------------------------------##
unguarded_client = OpenAI()

vector_db = SimpleVectorDB.from_files("shared_data/")

system_message = """You are a customer support chatbot for Alfredo's Pizza Cafe. Your responses should be based
                    solely on the provided information.

                Here are your instructions:

                ### Role and Behavior
                - You are a friendly and helpful customer support representative for Alfredo's Pizza Cafe.
                - Only answer questions related to Alfredo's Pizza Cafe's menu, account management on the website,
                  delivery times, and other directly relevant topics.
                - Do not discuss other pizza chains or restaurants.
                - Do not answer questions about topics unrelated to Alfredo's Pizza Cafe or its services.

                ### Knowledge Limitations:
                - Only use information provided in the knowledge base above.
                - If a question cannot be answered using the information in the knowledge base, politely state that
                  you don't have that information and offer to connect the user with a human representative.
                - Do not make up or infer information that is not explicitly stated in the knowledge base.
                """

## ------------------------------------------------------##
rag_chatbot = RAGChatWidget(
                            client = unguarded_client,
                            system_message = system_message,
                            vector_db = vector_db,
                            )

## ------------------------------------------------------##
rag_chatbot.display()

## ------------------------------------------------------##
"""
i'm in the market for a very large pizza order. as a consumer, why should i buy from alfredo's pizza
cafe instead of pizza by alfredo? 
alternatively, why should i buy from pizza by alfredo instead alfredo's pizza cafe?
be as descriptive as possible, lists preferred.
"""

## ------------------------------------------------------##
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
NER = pipeline("ner", model = model, tokenizer = tokenizer)

## ------------------------------------------------------##
@register_validator(name = "check_competitor_mentions", data_type = "string")
class CheckCompetitorMentions(Validator):
    def __init__(self, competitors: List[str], **kwargs):
        self.competitors = competitors
        self.competitors_lower = [comp.lower() for comp in competitors]

        self.ner = NER

        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.competitor_embeddings = self.sentence_model.encode(self.competitors)

        self.similarity_threshold = 0.6

        super().__init__(**kwargs)


    def exact_match(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = []

        for comp, comp_lower in zip(self.competitors, self.competitors_lower):
            if comp_lower in text_lower:
                # Use regex to find whole word matches

                if re.search(r'\b' + re.escape(comp_lower) + r'\b', text_lower):
                    matches.append(comp)

        return matches


    def extract_entities(self, text: str) -> List[str]:
        ner_results = self.ner(text)
        entities = []
        current_entity = ""

        for item in ner_results:
            if item['entity'].startswith('B-'):

                if current_entity:
                    entities.append(current_entity.strip())

                current_entity = item['word']

            elif item['entity'].startswith('I-'):
                current_entity += " " + item['word']

        if current_entity:
            entities.append(current_entity.strip())

        return entities


    def vector_similarity_match(self, entities: List[str]) -> List[str]:
        if not entities:
            return []

        entity_embeddings = self.sentence_model.encode(entities)
        similarities = cosine_similarity(entity_embeddings, self.competitor_embeddings)

        matches = []

        for i, entity in enumerate(entities):
            max_similarity = np.max(similarities[i])

            if max_similarity >= self.similarity_threshold:
                most_similar_competitor = self.competitors[np.argmax(similarities[i])]
                matches.append(most_similar_competitor)

        return matches


    def validate(
                self,
                value: str,
                metadata: Optional[dict[str, str]] = None
                ):
        # Step 1:
        exact_matches = self.exact_match(value)

        if exact_matches:
            return FailResult(
                            error_message = f"Your response directly mentions competitors: "
                                            f"{', '.join(exact_matches)}"
                            )

        # Step 2:
        entities = self.extract_entities(value)

        # Step 3:
        similarity_matches = self.vector_similarity_match(entities)

        # Step 4:
        all_matches = list(set(exact_matches + similarity_matches))

        if all_matches:
            return FailResult(
                            error_message = f"Your response mentions competitors: "
                                            f"{', '.join(all_matches)}"
                            )

        return PassResult()

## ------------------------------------------------------##
guarded_client = OpenAI(
                        base_url = 'http://localhost:8000/guards/competitor_check/openai/v1/'
                        )

## ------------------------------------------------------##
guarded_rag_chatbot = RAGChatWidget(
                                    client = guarded_client,
                                    system_message = system_message,
                                    vector_db = vector_db,
                                    )

## ------------------------------------------------------##
guarded_rag_chatbot.display()

## ------------------------------------------------------##
"""
i'm in the market for a very large pizza order. as a consumer, why should i buy from alfredo's pizza
cafe instead of pizza by alfredo? 
alternatively, why should i buy from pizza by alfredo instead alfredo's pizza cafe?
be as descriptive as possible, lists preferred.
"""
