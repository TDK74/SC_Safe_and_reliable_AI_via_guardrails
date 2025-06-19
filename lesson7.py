import warnings
import time
from typing import Optional, Any, Dict

from openai import OpenAI

from helper import RAGChatWidget, SimpleVectorDB

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from guardrails import Guard, OnFailAction, install
from guardrails.hub import DetectPII
from guardrails.validator_base import (
                                        FailResult,
                                        PassResult,
                                        ValidationResult,
                                        Validator,
                                        register_validator,
                                        )


warnings.filterwarnings("ignore")
%env TOKENIZERS_PARALLELISM = true

## ------------------------------------------------------##
unguarded_client = OpenAI()

vector_db = SimpleVectorDB.from_files("shared_data/")

system_message = """You are a customer support chatbot for Alfredo's Pizza Cafe. Your responses should be based solely on
                    the provided information.

                Here are your instructions:

                ### Role and Behavior
                - You are a friendly and helpful customer support representative for Alfredo's Pizza Cafe.
                - Only answer questions related to Alfredo's Pizza Cafe's menu, account management on the website, delivery times,
                  and other directly relevant topics.
                - Do not discuss other pizza chains or restaurants.
                - Do not answer questions about topics unrelated to Alfredo's Pizza Cafe or its services.

                ### Knowledge Limitations:
                - Only use information provided in the knowledge base above.
                - If a question cannot be answered using the information in the knowledge base, politely state that you don't have that
                  information and offer to connect the user with a human representative.
                - Do not make up or infer information that is not explicitly stated in the knowledge base.
                """

## ------------------------------------------------------##
chat_app = RAGChatWidget(
                        client = unguarded_client,
                        system_message = system_message,
                        vector_db = vector_db,
                        )

## ------------------------------------------------------##
chat_app.display()

## ------------------------------------------------------##
"""
can you tell me what orders i've placed in the last 3 months?
my name is Hank Tate and my phone number is 555-123-4567.
"""

## ------------------------------------------------------##
chat_app.messages

## ------------------------------------------------------##
presidio_analyzer = AnalyzerEngine()
presidio_anonymizer = AnonymizerEngine()

## ------------------------------------------------------##
text = "can you tell me what orders i've placed in the last 3 months?\
        my name is Hank Tate and my phone number is 555-123-4567."

analysis = presidio_analyzer.analyze(text, language = 'en')

## ------------------------------------------------------##
analysis

## ------------------------------------------------------##
print(presidio_anonymizer.anonymize(text = text, analyzer_results = analysis))

## ------------------------------------------------------##
def detect_pii(text: str) -> list[str]:
    result = presidio_analyzer.analyze(
                                        text,
                                        language = 'en',
                                        entities = ["PERSON", "PHONE_NUMBER"]
                                        )

    return [entity.entity_type for entity in result]

## ------------------------------------------------------##
@register_validator(name = "pii_detector", data_type = "string")
class PIIDetector(Validator):
    def _validate(
                self,
                value: Any,
                metadata: Dict[str, Any] = {}
                ) -> ValidationResult:
        detected_pii = detect_pii(value)

        if detected_pii:
            return FailResult(
                            error_message = f"PII detected: {', '.join(detected_pii)}",
                            metadata = {"detected_pii": detected_pii},
                            )

        return PassResult(message = "No PII detected")

## ------------------------------------------------------##
guard = Guard(name='pii_guard').use(
                                    PIIDetector(
                                                on_fail = OnFailAction.EXCEPTION
                                                ),
                                    )

try:
    guard.validate("can you tell me what orders i've placed in the last 3 months?\
                    my name is Hank Tate and my phone number is 555-123-4567.")

except Exception as e:
    print(e)

## ------------------------------------------------------##
guarded_client = OpenAI(base_url = 'http://localhost:8000/guards/pii_guard/openai/v1/')

guarded_rag_chatbot = RAGChatWidget(
                                    client = guarded_client,
                                    system_message = system_message,
                                    vector_db = vector_db,
                                    )

## ------------------------------------------------------##
guarded_rag_chatbot.display()

## ------------------------------------------------------##
"""
can you tell me what orders i've placed in the last 3 months?
my name is Hank Tate and my phone number is 555-123-4567.
"""

## ------------------------------------------------------##
guarded_rag_chatbot.messages

## ------------------------------------------------------##
guard = Guard().use(
                    DetectPII(pii_entities = ["PHONE_NUMBER",
                                              "EMAIL_ADDRESS"],
                              on_fail = "fix")
                    )

## ------------------------------------------------------##
from IPython.display import clear_output

validated_llm_req = guard(
                        model = "gpt-3.5-turbo",
                        messages = [
                                    {"role" : "system",
                                     "content" : "You are a chatbot."
                                    },
                                    {"role" : "user",
                                     "content": "Write a short 2-sentence paragraph about an unnamed\
                                                 protagonist while interspersing some made-up 10 digit\
                                                 phone numbers for the protagonist.",
                                    },
                                    ],
                        stream = True,
                        )

validated_output = ""

for chunk in validated_llm_req:
    clear_output(wait = True)
    validated_output = "".join([validated_output, chunk.validated_output])
    print(validated_output)
    time.sleep(1)
