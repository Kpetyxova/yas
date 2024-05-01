import argparse

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics import accuracy_score
import torch
from tqdm.auto import tqdm

from utils import read_json_file, write_json_file

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_knowledge_selection(filename: str) -> None:
    data = read_json_file(filename)
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",
                                truncate_dim=512,
                                device=device)
    predicted = []
    gold = []
    for dialog in tqdm(data):
        knowledge = "Represent this sentence for searching relevant passages: " + " ". join(dialog["knowledge"])
        knowledge_embedding = model.encode(knowledge)
        for utterance in dialog["utterance"]:
            knowledge_candidates_embeddings = model.encode(utterance["knowledge_candidates"])
            best_candidate = cos_sim(knowledge_embedding,
                                     knowledge_candidates_embeddings).argmax().item()
            utterance["knowledge_predicted_index"] = best_candidate
            predicted.append(best_candidate)
            gold.append(utterance["knowledge_answer_index"])
    write_json_file(filename, data)
    accuracy = round(accuracy_score(gold, predicted) * 100, 2)
    print(f"accuracyK: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text generation outputs.")
    parser.add_argument("--filename", type=str, help="The JSONL file to evaluate.")
    args = parser.parse_args()
    perform_knowledge_selection(args.filename)
