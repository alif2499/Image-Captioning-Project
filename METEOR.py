import json
from nltk.translate.meteor_score import meteor_score

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

meteor_scores = []

for image_id, predicted_caption in predictions.items():
    # Get ground truth captions for the current image
    reference_captions = ground_truth_preprocessed[image_id]

    # Tokenize the predicted caption (assuming it's a space-separated string)
    predicted_tokens = predicted_caption.split()

    # Compute METEOR score for the predicted caption against each reference caption
    image_meteor_scores = []
    for reference_caption in reference_captions:
        reference_tokens = reference_caption.split()
        # Compute METEOR score for the current reference caption
        score = meteor_score([reference_tokens], predicted_tokens)
        image_meteor_scores.append(score)
    
    # Average METEOR scores for all reference captions of the current image
    average_meteor_score = sum(image_meteor_scores) / len(image_meteor_scores)
    meteor_scores.append(average_meteor_score)

# Calculate average METEOR score
average_meteor_score = sum(meteor_scores) / len(meteor_scores)

print(f"Average METEOR score: {average_meteor_score:.4f}")


# import json
# from nltk.translate.meteor_score import meteor_score

# # Load ground truth and predicted captions from JSON files
# def load_json(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data

# ground_truth_file = 'dummy_test.json'
# prediction_file = 'prediction.json'

# ground_truth = load_json(ground_truth_file)
# predictions = load_json(prediction_file)

# def preprocess_captions(captions_list):
#     return [caption.replace('sos ', '').replace(' eos', '') for caption in captions_list]

# # Preprocess ground truth captions
# ground_truth_preprocessed = {image_id: preprocess_captions(captions) for image_id, captions in ground_truth.items()}

# meteor_scores = []

# for image_id, predicted_caption in predictions.items():
#     # Get ground truth captions for the current image
#     reference_captions = ground_truth_preprocessed[image_id]

#     # Tokenize the predicted caption (hypothesis) into a list of words
#     predicted_tokens = predicted_caption.split()

#     # Calculate METEOR score for the predicted caption against all reference captions
#     scores = [meteor_score([ref_caption.split()], predicted_tokens) for ref_caption in reference_captions]

#     # Take the maximum METEOR score from all reference captions for the current image
#     max_score = max(scores)
#     meteor_scores.append(max_score)

# # Calculate average METEOR score
# average_meteor_score = sum(meteor_scores) / len(meteor_scores)

# print(f"Average METEOR score: {average_meteor_score:.4f}")
