import json
import os

def parse_frames_dataset(json_file_path):
    """
    Parses the Microsoft Research Frames JSON file to build flat structural logs
    containing user text, intent classifications, and mapped frame slots.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: Target dataset file not found at {json_file_path}")
        return []

    parsed_training_data = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        dialogues = json.load(file)

    for dialogue in dialogues:
        dialogue_id = dialogue.get("id", "unknown_id")
        
        # Traverse each individual turn in the conversation
        for turn_idx, turn in enumerate(dialogue.get("turns", [])):
            author = turn.get("author")
            text = turn.get("text", "")
            
            # Initialize storage containers for intents and active slot tracking
            intents = []
            extracted_slots = {}
            
            labels = turn.get("labels", {})
            acts = [act.get("name") for act in labels.get("acts", [])]
            
            # Process frames tracking if they exist inside this turn's metadata
            if "frames" in labels:
                for frame in labels["frames"]:
                    info = frame.get("info", {})
                    
                    # Extract high-level intents
                    if "intent" in info:
                        intents.extend(info["intent"])
                        
                    # Extract variable slots (e.g. destination, budget constraints)
                    for slot_name, slot_values in info.items():
                        if slot_name != "intent" and isinstance(slot_values, list):
                            # The dataset structures slot values as a list of dict objects
                            extracted_values = [v.get("val") for v in slot_values if "val" in v]
                            if extracted_values:
                                extracted_slots[slot_name] = extracted_values

            # Append the structured turn dictionary directly into our training pool
            parsed_training_data.append({
                "dialogue_id": dialogue_id,
                "turn_id": turn_idx,
                "user_role": author,
                "utterance": text,
                "dialogue_acts": acts,
                "intents": intents,
                "slots": extracted_slots
            })

    return parsed_training_data

# --- Example Usage Outline ---
if __name__ == "__main__":
    # Path to where you saved the unzipped frames.json file
    dataset_path = "data/frames.json"
    
    # Run the execution pipeline
    cleaned_data = parse_frames_dataset(dataset_path)
    
    # Save a small subset to inspect the training structure
    if cleaned_data:
        print(f"Successfully processed {len(cleaned_data)} turn elements!")
        print("\nSample Turn Profile Output:")
        print(json.dumps(cleaned_data[0], indent=4))
