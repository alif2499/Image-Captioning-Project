import json
from rouge_score import rouge_scorer

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_caption(caption):
    # Remove "sos" and "eos" markers and strip any extra spaces
    caption = caption.replace('sos ', '').replace(' eos', '').strip()
    return caption

ground_truth_file = 'COCO_dataset\\test.json'
prediction_file = 'prediction.json'

# Load ground truth and predicted captions
ground_truth = load_json(ground_truth_file)
predictions = load_json(prediction_file)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Lists to store individual ROUGE scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Iterate over each image and its predicted caption
for image_id, predicted_caption in predictions.items():
    # Get the list of ground truth captions for the current image
    reference_captions = ground_truth[image_id]

    # Preprocess the predicted caption (remove leading/trailing spaces)
    predicted_caption = predicted_caption.strip()

    # Calculate ROUGE scores for each reference caption against the predicted caption
    for ref_caption in reference_captions:
        # Preprocess the reference caption to remove "sos" and "eos" markers
        ref_caption = preprocess_caption(ref_caption)

        # Calculate ROUGE scores
        scores = scorer.score(ref_caption, predicted_caption)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Calculate average ROUGE scores across all reference captions
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Print or use the overall ROUGE scores
print("Overall ROUGE Scores:")
print(f"ROUGE-1: {average_rouge1:.4f}\nROUGE-2: {average_rouge2:.4f}\nROUGE-L: {average_rougeL:.4f}")
