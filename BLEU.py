import json
from nltk.translate.bleu_score import sentence_bleu

# Load ground truth and predicted captions from JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

ground_truth_file = 'COCO_dataset\\test.json'
prediction_file = 'prediction.json'

ground_truth = load_json(ground_truth_file)
predictions = load_json(prediction_file)

def preprocess_captions(captions_list):
    return [caption.replace('sos ', '').replace(' eos', '') for caption in captions_list]

# Preprocess ground truth captions
ground_truth_preprocessed = {image_id: preprocess_captions(captions) for image_id, captions in ground_truth.items()}

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

for image_id, predicted_caption in predictions.items():
    # Get ground truth captions for the current image
    reference_captions = ground_truth_preprocessed[image_id]

    # Tokenize the predicted caption (assuming it's a space-separated string)
    predicted_tokens = predicted_caption.split()

    # Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for the predicted caption against each reference caption
    image_bleu1_scores = []
    image_bleu2_scores = []
    image_bleu3_scores = []
    image_bleu4_scores = []

    for reference_caption in reference_captions:
        reference_tokens = reference_caption.split()
        
        # Compute BLEU-1 score for the current reference caption
        bleu1_score = sentence_bleu([reference_tokens], predicted_tokens, weights=(1, 0, 0, 0))
        image_bleu1_scores.append(bleu1_score)

        # Compute BLEU-2 score for the current reference caption
        bleu2_score = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.5, 0.5, 0, 0))
        image_bleu2_scores.append(bleu2_score)

        # Compute BLEU-3 score for the current reference caption
        bleu3_score = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.33, 0.33, 0.33, 0))
        image_bleu3_scores.append(bleu3_score)

        # Compute BLEU-4 score for the current reference caption
        bleu4_score = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        image_bleu4_scores.append(bleu4_score)
    
    # Average BLEU scores for all reference captions of the current image
    average_bleu1_score = sum(image_bleu1_scores) / len(image_bleu1_scores)
    average_bleu2_score = sum(image_bleu2_scores) / len(image_bleu2_scores)
    average_bleu3_score = sum(image_bleu3_scores) / len(image_bleu3_scores)
    average_bleu4_score = sum(image_bleu4_scores) / len(image_bleu4_scores)

    bleu1_scores.append(average_bleu1_score)
    bleu2_scores.append(average_bleu2_score)
    bleu3_scores.append(average_bleu3_score)
    bleu4_scores.append(average_bleu4_score)

# Calculate average BLEU scores across all predicted captions
average_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
average_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
average_bleu3 = sum(bleu3_scores) / len(bleu3_scores)
average_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

# Print or use the average BLEU scores
print("Average BLEU Scores:")
print(f"BLEU-1: {average_bleu1:.4f}")
print(f"BLEU-2: {average_bleu2:.4f}")
print(f"BLEU-3: {average_bleu3:.4f}")
print(f"BLEU-4: {average_bleu4:.4f}")
