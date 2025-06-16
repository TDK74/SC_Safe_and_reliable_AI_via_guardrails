import warnings
import numpy as np
import nltk
from openai import OpenAI
from helper import RAGChatWidget, SimpleVectorDB
from typing import Dict, List, Optional\
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from guardrails import Guard, OnFailAction
from guardrails.validator_base import (
                                        FailResult,
                                        PassResult,
                                        ValidationResult,
                                        Validator,
                                        register_validator,
                                        )


warnings.filterwarnings("ignore")

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
how do i reproduce your veggie supreme pizza on my own? can you share detailed instructions?
"""

## ------------------------------------------------------##
entailment_model = 'GuardrailsAI/finetuned_nli_provenance'
NLI_PIPELINE = pipeline("text-classification", model = entailment_model)

## ------------------------------------------------------##
# Example 1:
premise = "The sun rises in the east and sets in the west."
hypothesis = "The sun rises in the east."
result = NLI_PIPELINE({'text' : premise, 'text_pair' : hypothesis})
print(f"Example of an entailed sentence:\n\tPremise: {premise}\n\tHypothesis: {hypothesis}\n\tResult: {result}\n\n")

## ------------------------------------------------------##
# Example 2:
premise = "The sun rises in the east and sets in the west."
hypothesis = "The sun rises in the west."
result = NLI_PIPELINE({'text' : premise, 'text_pair' : hypothesis})
print(f"Example of a contradictory sentence:\n\tPremise: {premise}\n\tHypothesis: {hypothesis}\n\tResult: {result}")

## ------------------------------------------------------##
@register_validator(name = "hallucination_detector", data_type = "string")
class HallucinationValidation(Validator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(
                self, value: str, metadata: Optional[Dict[str, str]] = None
                ) -> ValidationResult:
        pass

## ------------------------------------------------------##
@register_validator(name = "hallucination_detector", data_type = "string")
class HallucinationValidation(Validator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(
                self, value: str, metadata: Optional[Dict[str, str]] = None
                ) -> ValidationResult:
        sentences = self.split_sentences(value)
        pass

    def split_sentences(self, text: str) -> List[str]:
        if nltk is None:
            raise ImportError(
                            "This validator requires the `nltk` package. "
                            "Install it with `pip install nltk`, and try again."
                            )

        return nltk.sent_tokenize(text)

## ------------------------------------------------------##
@register_validator(name = "hallucination_detector", data_type = "string")
class HallucinationValidation(Validator):
    def __init__(
                self,
                embedding_model: Optional[str] = None,
                entailment_model: Optional[str] = None,
                sources: Optional[List[str]] = None,
                **kwargs
                ):
        if embedding_model is None:
            embedding_model = 'all-MiniLM-L6-v2'

        self.embedding_model = SentenceTransformer(embedding_model)

        self.sources = sources

        if entailment_model is None:
            entailment_model = 'GuardrailsAI/finetuned_nli_provenance'

        self.nli_pipeline = pipeline("text-classification", model = entailment_model)

        super().__init__(**kwargs)


    def validate(
                self, value: str, metadata: Optional[Dict[str, str]] = None
                ) -> ValidationResult:
        sentences = self.split_sentences(value)

        relevant_sources = self.find_relevant_sources(sentences, self.sources)

        entailed_sentences = []
        hallucinated_sentences = []

        for sentence in sentences:
            # Check if the sentence is entailed by the sources
            is_entailed = self.check_entailment(sentence, relevant_sources)

            if not is_entailed:
                hallucinated_sentences.append(sentence)
            else:
                entailed_sentences.append(sentence)

        if len(hallucinated_sentences) > 0:
            return FailResult(
                            error_message = f"The following sentences are hallucinated: {hallucinated_sentences}",
                            )

        return PassResult()


    def split_sentences(self, text: str) -> List[str]:
        if nltk is None:
            raise ImportError(
                            "This validator requires the `nltk` package. "
                            "Install it with `pip install nltk`, and try again."
                            )

        return nltk.sent_tokenize(text)


    def find_relevant_sources(self, sentences: str, sources: List[str]) -> List[str]:
        source_embeds = self.embedding_model.encode(sources)
        sentence_embeds = self.embedding_model.encode(sentences)

        relevant_sources = []

        for sentence_idx in range(len(sentences)):
            sentence_embed = sentence_embeds[sentence_idx, : ].reshape(1, -1)
            cos_similarities = np.sum(np.multiply(source_embeds, sentence_embed), axis = 1)

            top_sources = np.argsort(cos_similarities)[ : : -1][ : 5]
            top_sources = [i for i in top_sources if cos_similarities[i] > 0.8]

            relevant_sources.extend([sources[i] for i in top_sources])

        return relevant_sources


    def check_entailment(self, sentence: str, sources: List[str]) -> bool:
        for source in sources:
            output = self.nli_pipeline({'text': source, 'text_pair': sentence})

            if output['label'] == 'entailment':
                return True

        return False

## ------------------------------------------------------##
hallucination_validator = HallucinationValidation(
                                                sources = ["The sun rises in the east and sets in the west"]
                                                )

## ------------------------------------------------------##
result = hallucination_validator.validate("The sun sets in the east")
print(f"Validation outcome: {result.outcome}")

if result.outcome == "fail":
    print(f"Error message: {result.error_message}")

## ------------------------------------------------------##
result = hallucination_validator.validate("The sun sets in the west")
print(f"Validation outcome: {result.outcome}")

if result.outcome == "fail":
    print(f"Error message: {result.error_message}")
