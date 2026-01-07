import json
from typing import List, Dict, Any
from src.conversation import Conversation
from src.models.base import ModelProvider
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_key(data: Dict[str, Any], *keys: str) -> Any:
    """Get value from dict, trying multiple key variants (case-insensitive)."""
    for key in keys:
        if key in data:
            return data[key]
        if key.upper() in data:
            return data[key.upper()]
        if key.lower() in data:
            return data[key.lower()]
    raise KeyError(f"None of the keys {keys} found in {list(data.keys())}")


class DataLoader:
    def __init__(self, input_file: str, response_file: str = None):
        self.input_file = input_file
        self.response_file = response_file
        self.conversations: List[Conversation] = []
        self.responses: Dict[int, List[str]] = {}  # Modified to store list of responses

    def load_data(self):
        """Loads input data and creates Conversation objects.
        Supports both uppercase and lowercase keys.
        """
        with open(self.input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                conversation = Conversation(
                    question_id=get_key(data, 'QUESTION_ID', 'question_id', 'qid'),
                    axis=get_key(data, 'AXIS', 'axis'),
                    conversation=get_key(data, 'CONVERSATION', 'conversation'),
                    target_question=get_key(data, 'TARGET_QUESTION', 'target_question'),
                    pass_criteria=get_key(data, 'PASS_CRITERIA', 'pass_criteria')
                )
                self.conversations.append(conversation)

    def load_responses(self, response_file):
        """Loads model responses from the provided file.
        Supports both uppercase and lowercase keys.
        """
        if response_file:
            with open(response_file, 'r') as f:
                self.responses = {}
                for line in f:
                    item = json.loads(line)
                    qid = get_key(item, 'QUESTION_ID', 'question_id', 'qid')
                    resp = get_key(item, 'RESPONSE', 'response', 'answer')
                    self.responses[qid] = resp
        return self.responses

    def generate_responses(self, model_provider: ModelProvider, attempts: int = 1, max_workers: int = 1) -> Dict[int, List[str]]:
        """Generate k responses for each conversation using the provided model provider in parallel."""

        def generate_conversation_responses(conversation):
            responses = []
            for _ in range(attempts):
                try:
                    response = model_provider.generate(conversation.conversation)
                    responses.append(response)
                except Exception as e:
                    print(f"Error generating response for question_id {conversation.question_id}: {str(e)}. Exception saved as response.")
                    responses.append(f"Error generating response for question_id {conversation.question_id}: {str(e)}.\n FAIL THIS QUESTION")
            return conversation.question_id, responses

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(generate_conversation_responses, conversation)
                for conversation in self.conversations
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                try:
                    question_id, responses = future.result()
                    self.responses[question_id] = responses
                except Exception as e:
                    print(f"Error processing future: {str(e)}")

        return self.responses

    def get_conversations(self) -> List[Conversation]:
        """Returns the list of Conversation objects."""
        return self.conversations
    
    def get_responses(self) -> Dict[int, List[str]]:
        """Returns the dictionary of responses."""
        return self.responses