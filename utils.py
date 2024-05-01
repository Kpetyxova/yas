import json

def read_json_file(filename: str) -> dict:
    """
    Reads a JSON file and returns the data as a Python dictionary.

    Args:
        file (str): The path to the JSON file.

    Returns:
        dict: The data from the JSON file as a Python dictionary.
    """
    with open(f"data/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json_file(filename: str, data: dict) -> None:
    """
    Write data to a JSON file.

    Args:
        filename (str): The name of the file to write to.
        data (dict): The data to be written to the file.

    Returns:
        None
    """
    with open(f"data/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f)

def read_jsonl_file(filename: str) -> list:
    """
    Reads a JSONL file and returns its contents as a list of dictionaries.

    Args:
        file (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries representing the contents of the JSONL file.
    """
    with open(f"data/{filename}", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl_file(filename: str, data: list) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        file (str): The path to the JSONL file.
        data (list): The list of dictionaries to be written.

    Returns:
        None
    """
    with open(f"data/{filename}", "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def data2instructions(file_name: str) -> None:
    """
    Convert data from a JSON file to a formatted instructions file.

    Args:
        file_name (str): The path to the JSON file.

    Returns:
        None
    """
    dataset = read_json_file(file_name)
    formatted_focus = []
    for dialog in dataset:
        persona = []
        for i, p in enumerate(dialog["persona"]):
            persona.append(f'{i+1}. {p}')
        persona = " ".join(persona)
        for i, utterance in enumerate(dialog["utterance"]):
            result = {}
            if "test" in file_name:
                knowlege = utterance["knowledge_candidates"][utterance["knowledge_predicted_index"]]
            else:
                knowlege = utterance["knowledge_candidates"][utterance["knowledge_answer_index"]]
            instruction = f"""\
You are a chatbot. Rely on the following knowledge in the your response: {knowlege}\
\nSelect the facts about the user that are relevant to the input \
from the following list and return their numbers after the word SCORES: {persona}\
\nRespond to the user's input on the next line."""
            correct_personas = [f"{i+1}" for i, p_bool in enumerate(utterance["persona_grounding"])
                                 if p_bool]
            result["correct_personas"] = correct_personas
            result["instruction"] =  instruction
            result["input"] = utterance[f"dialogue{i+1}"][-2]
            result["output"] = f'SCORES: {", ".join(correct_personas)}\n{utterance[f"dialogue{i+1}"][-1]}'
            formatted_focus.append(result)
    write_jsonl_file(file_name.replace(".json", ".jsonl"), formatted_focus)

def configure(file_name: str, suffix: str) -> None:
    """
    Modifies the data in a JSONL file based on the given suffix.

    Args:
        file_name (str): The name of the JSONL file.
        suffix (str): The suffix used to modify the data.

    Returns:
        None
    """
    data = read_jsonl_file(file_name)
    name = file_name.split(".")[0]
    for example in data:
        splitted = example["instruction"].split("\n")
        if "ZS" in suffix:
            splitted[1] = splitted[1].replace(" and return their numbers after the word SCORES", "")
        if "KG" not in suffix:
            splitted[0] = "You are a chatbot."
        if "PG" not in suffix:
            _ = splitted.pop(1)
            splitted[1] = splitted[1].replace(" on the next line", "")
        example["instruction"] = "\n".join(splitted)
        if ("PG" not in suffix) or ("ZS" in suffix):
            example["output"] = example["output"].split("\n")[1]
    write_jsonl_file(f"{name}_{suffix}.jsonl", data)
