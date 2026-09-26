#%% [1] import

import torch
from transformers import AutoTokenizer, AutoModel

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
#%% [2] BERTのロード

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)

print("BERT loaded")
#%% [3] Device

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

bert = bert.to(device)

print("Device:", device)
#%% [4] Tokenize

text = "大阪の府庁所在地は？"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("input_ids:")
print(inputs["input_ids"])

print("input_ids shape:")
print(inputs["input_ids"].shape)

#%% [5] Move inputs to device

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("input_ids device:", inputs["input_ids"].device)
#%% [6] BERT forward

with torch.no_grad():
    outputs = bert(**inputs)

print(type(outputs))
print("last_hidden_state shape:")
print(outputs.last_hidden_state.shape)
#%% [7] Extract CLS

cls_embedding = outputs.last_hidden_state[:, 0, :]

print("CLS shape:")
print(cls_embedding.shape)

#%% [8] Decision Head

import torch.nn as nn

classifier = nn.Linear(768, 3)
classifier = classifier.to(device)

logits = classifier(cls_embedding)

print("Logits:")
print(logits)

print("Logits shape:")
print(logits.shape)
#%% [9] Softmax

probabilities = torch.softmax(logits, dim=-1)

print("Probabilities:")
print(probabilities)

print("Probability sum:")
print(probabilities.sum())
print(cls_embedding.shape)
print(logits.shape)
print(probabilities.shape)
#%% [10] Loss function

criterion = nn.CrossEntropyLoss()

print(criterion)
#%% [11] Correct label

label = torch.tensor([0], device=device)

print("Label:")
print(label)

print("Label shape:")
print(label.shape)
#%% [12] Calculate loss

loss = criterion(logits, label)

print("Loss:")
print(loss)

print("Loss value:")
print(loss.item())
#%% [13] Optimizer

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3
)

print(optimizer)
#%% [14] Backward

optimizer.zero_grad()

loss.backward()

print("Backward completed")
#%% [15] Update parameters

optimizer.step()

print("Parameters updated")
#%% [16] Check prediction after update

with torch.no_grad():
    logits_after = classifier(cls_embedding)
    probabilities_after = torch.softmax(logits_after, dim=-1)

print("Logits after:")
print(logits_after)

print("Probabilities after:")
print(probabilities_after)

print("Probability sum:")
print(probabilities_after.sum())
#%% [17] Check classifier

print(classifier)
#%% [18] Check classifier parameters

print("Weight shape:")
print(classifier.weight.shape)

print("Bias shape:")
print(classifier.bias.shape)
#%% [19] Training data

texts = [
    "日本の首都はどこですか？",
    "日本の首都を教えてください。",
    "大阪府の府庁所在地は？",
    "大阪の府庁所在地を教えてください。",
    "京都府の府庁所在地は？",
    "京都の府庁所在地を教えてください。"
]

labels = torch.tensor(
    [0, 0, 1, 1, 2, 2],
    device=device
)

print("Number of texts:", len(texts))
print("Labels:", labels)
#%% [20] Tokenize training data

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("input_ids shape:")
print(inputs["input_ids"].shape)
#%% [21] BERT forward

with torch.no_grad():
    outputs = bert(**inputs)

print("last_hidden_state shape:")
print(outputs.last_hidden_state.shape)
#%% [22] Extract CLS embeddings

cls_embeddings = outputs.last_hidden_state[:, 0, :]

print("CLS embeddings shape:")
print(cls_embeddings.shape)
#%% [23] Decision Head

logits = classifier(cls_embeddings)

print("Logits:")
print(logits)

print("Logits shape:")
print(logits.shape)
#%% [24] Softmax

probabilities = torch.softmax(logits, dim=-1)

print("Probabilities:")
print(probabilities)

print("Probability sums:")
print(probabilities.sum(dim=-1))
#%% [25] Calculate batch loss

loss = criterion(logits, labels)

print("Loss:")
print(loss)

print("Loss value:")
print(loss.item())
#%% [26] Backward

optimizer.zero_grad()

loss.backward()

print("Backward completed")
#%% [27] Update parameters

optimizer.step()

print("Parameters updated")
#%% [28] Check predictions after update

with torch.no_grad():
    logits_after = classifier(cls_embeddings)
    probabilities_after = torch.softmax(logits_after, dim=-1)

print("Probabilities after:")
print(probabilities_after)

print("Predicted classes:")
print(probabilities_after.argmax(dim=-1))

print("Correct labels:")
print(labels)
#%% [29] Phase 2 - State and Question

state = "日本についての情報"
question = "首都はどこですか？"

text = f"{state}。{question}"

print("State:")
print(state)

print("Question:")
print(question)

print("Combined text:")
print(text)
#%% [30] Tokenize State + Question

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("input_ids:")
print(inputs["input_ids"])

print("input_ids shape:")
print(inputs["input_ids"].shape)

#%% [31] BERT forward

with torch.no_grad():
    outputs = bert(**inputs)

print("last_hidden_state shape:")
print(outputs.last_hidden_state.shape)
#%% [32] Extract CLS embedding

cls_embedding = outputs.last_hidden_state[:, 0, :]

print("CLS embedding:")
print(cls_embedding)

print("CLS embedding shape:")
print(cls_embedding.shape)

#%% [33] Decision Head

logits = classifier(cls_embedding)

print("Logits:")
print(logits)

print("Logits shape:")
print(logits.shape)

#%% [34] Softmax

probabilities = torch.softmax(logits, dim=-1)

print("Probabilities:")
print(probabilities)

print("Probability sum:")
print(probabilities.sum())
#%% [35] Change Question

state = "日本についての情報"
question = "最大の都市はどこですか？"

text = f"{state}。{question}"

print("Combined text:")
print(text)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = bert(**inputs)

cls_embedding_2 = outputs.last_hidden_state[:, 0, :]

print("CLS embedding shape:")
print(cls_embedding_2.shape)
#%% [36] Decision Head with new Question

logits_2 = classifier(cls_embedding_2)

probabilities_2 = torch.softmax(logits_2, dim=-1)

print("Logits:")
print(logits_2)

print("Probabilities:")
print(probabilities_2)

print("Probability sum:")
print(probabilities_2.sum())
#%% [37] Test another Question

state = "日本についての情報"
question = "通貨は何ですか？"

text = f"{state}。{question}"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = bert(**inputs)

cls_embedding_3 = outputs.last_hidden_state[:, 0, :]

logits_3 = classifier(cls_embedding_3)
probabilities_3 = torch.softmax(logits_3, dim=-1)

print("Question:")
print(question)

print("Probabilities:")
print(probabilities_3)

print("Predicted class:")
print(probabilities_3.argmax(dim=-1))

#%% [38] Dynamic Choice - Data Structure

state = "日本についての情報"

question = "通貨は何ですか？"

choices = [
    "円",
    "ドル",
    "ユーロ"
]

answer = "円"

print("State:")
print(state)

print("\nQuestion:")
print(question)

print("\nChoices:")
print(choices)

print("\nCorrect answer:")
print(answer)
#%% [39] Create Question + Choice pairs

choice_texts = [
    f"{question} {choice}"
    for choice in choices
]

print("Choice texts:")

for i, choice_text in enumerate(choice_texts):
    print(f"{i}: {choice_text}")
#%% [40] Tokenize Question + Choices

choice_inputs = tokenizer(
    choice_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

choice_inputs = {
    key: value.to(device)
    for key, value in choice_inputs.items()
}

print("input_ids shape:")
print(choice_inputs["input_ids"].shape)
#%% [41] BERT forward for each Choice

with torch.no_grad():
    choice_outputs = bert(**choice_inputs)

print("last_hidden_state shape:")
print(choice_outputs.last_hidden_state.shape)
#%% [42] Extract CLS for each Choice

choice_cls_embeddings = choice_outputs.last_hidden_state[:, 0, :]

print("Choice CLS embeddings shape:")
print(choice_cls_embeddings.shape)
#%% [43] Create State + Question + Choice pairs

choice_texts = [
    f"{state}。{question} {choice}"
    for choice in choices
]

print("Choice texts:")

for i, choice_text in enumerate(choice_texts):
    print(f"{i}: {choice_text}")
    #%% [44] BERT forward for State + Question + Choice

choice_inputs = tokenizer(
    choice_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

choice_inputs = {
    key: value.to(device)
    for key, value in choice_inputs.items()
}

with torch.no_grad():
    choice_outputs = bert(**choice_inputs)

print("last_hidden_state shape:")
print(choice_outputs.last_hidden_state.shape)
#%% [45] Extract CLS for each Choice

choice_cls_embeddings = choice_outputs.last_hidden_state[:, 0, :]

print("Choice CLS embeddings shape:")
print(choice_cls_embeddings.shape)
#%% [46] Choice Scoring Head

choice_scorer = nn.Linear(768, 1)
choice_scorer = choice_scorer.to(device)

print(choice_scorer)
print("Weight shape:")
print(choice_scorer.weight.shape)

print("Bias shape:")
print(choice_scorer.bias.shape)
#%% [47] Score each Choice

choice_scores = choice_scorer(choice_cls_embeddings)

print("Choice scores:")
print(choice_scores)

print("Choice scores shape:")
print(choice_scores.shape)
#%% [48] Softmax over Choices

choice_scores = choice_scores.squeeze(-1)

choice_probabilities = torch.softmax(
    choice_scores,
    dim=-1
)

print("Choice scores:")
print(choice_scores)

print("\nChoice probabilities:")
print(choice_probabilities)

print("\nProbability sum:")
print(choice_probabilities.sum())
#%% [49] Select predicted Choice

predicted_index = choice_probabilities.argmax(dim=-1)

predicted_choice = choices[predicted_index.item()]

print("Predicted index:")
print(predicted_index)

print("Predicted choice:")
print(predicted_choice)

print("Correct answer:")
print(answer)
#%% [50] Correct Choice Label

choice_label = torch.tensor(
    [choices.index(answer)],
    device=device
)

print("Choices:")
print(choices)

print("Correct answer:")
print(answer)

print("Correct choice index:")
print(choice_label)
#%% [51] Calculate Dynamic Choice Loss

loss = criterion(
    choice_scores.unsqueeze(0),
    choice_label
)

print("Choice scores:")
print(choice_scores)

print("Correct choice index:")
print(choice_label)

print("Loss:")
print(loss)

print("Loss value:")
print(loss.item())
#%% [52] Choice Scorer Optimizer

choice_optimizer = torch.optim.Adam(
    choice_scorer.parameters(),
    lr=1e-3
)

print(choice_optimizer)
#%% [53] Train One Step

choice_optimizer.zero_grad()

loss.backward()

choice_optimizer.step()

print("Choice scorer updated")
#%% [54] Check Prediction After Update

with torch.no_grad():
    choice_scores_after = choice_scorer(
        choice_cls_embeddings
    ).squeeze(-1)

    choice_probabilities_after = torch.softmax(
        choice_scores_after,
        dim=-1
    )

    predicted_index_after = choice_probabilities_after.argmax(
        dim=-1
    )

    predicted_choice_after = choices[
        predicted_index_after.item()
    ]

print("Choice scores after:")
print(choice_scores_after)

print("\nChoice probabilities after:")
print(choice_probabilities_after)

print("\nPredicted choice:")
print(predicted_choice_after)

print("\nCorrect answer:")
print(answer)
#%% [55] Dynamic Choice Training Data

dynamic_data = [
    {
        "state": "日本についての情報",
        "question": "首都はどこですか？",
        "choices": ["東京", "大阪", "京都"],
        "answer": "東京"
    },
    {
        "state": "日本についての情報",
        "question": "通貨は何ですか？",
        "choices": ["円", "ドル", "ユーロ"],
        "answer": "円"
    },
    {
        "state": "日本についての情報",
        "question": "最大の都市はどこですか？",
        "choices": ["東京", "大阪", "京都"],
        "answer": "東京"
    },
    {
        "state": "日本についての情報",
        "question": "大阪府の府庁所在地は？",
        "choices": ["大阪市", "京都市", "神戸市"],
        "answer": "大阪市"
    },
    {
        "state": "日本についての情報",
        "question": "京都府の府庁所在地は？",
        "choices": ["大阪市", "京都市", "奈良市"],
        "answer": "京都市"
    },
    {
        "state": "日本についての情報",
        "question": "日本の国会はどこにありますか？",
        "choices": ["東京", "大阪", "名古屋"],
        "answer": "東京"
    }
]

print("Number of training examples:", len(dynamic_data))

for i, item in enumerate(dynamic_data):
    print(f"\nExample {i}")
    print("Question:", item["question"])
    print("Choices:", item["choices"])
    print("Answer:", item["answer"])
#%% [56] Dynamic Choice Scoring Function

def score_choices(state, question, choices):
    choice_texts = [
        f"{state}。{question} {choice}"
        for choice in choices
    ]

    inputs = tokenizer(
        choice_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = bert(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    scores = choice_scorer(
        cls_embeddings
    ).squeeze(-1)

    return scores


print("Function created")
#%% [57] Dynamic Choice Training Function

def train_one_example(item):
    state = item["state"]
    question = item["question"]
    choices = item["choices"]
    answer = item["answer"]

    choice_texts = [
        f"{state}。{question} {choice}"
        for choice in choices
    ]

    inputs = tokenizer(
        choice_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = bert(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    scores = choice_scorer(
        cls_embeddings
    ).squeeze(-1)

    correct_index = choices.index(answer)

    label = torch.tensor(
        [correct_index],
        device=device
    )

    loss = criterion(
        scores.unsqueeze(0),
        label
    )

    choice_optimizer.zero_grad()
    loss.backward()
    choice_optimizer.step()

    return loss.item()
#%% [58] Train Dynamic Choice Model

num_epochs = 20

for epoch in range(num_epochs):

    total_loss = 0.0

    for item in dynamic_data:
        loss_value = train_one_example(item)
        total_loss += loss_value

    average_loss = total_loss / len(dynamic_data)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Average Loss: {average_loss:.4f}"
    )
#%% [59] Evaluate Dynamic Choice Model

correct = 0
total = len(dynamic_data)

for item in dynamic_data:

    scores = score_choices(
        item["state"],
        item["question"],
        item["choices"]
    )

    probabilities = torch.softmax(
        scores,
        dim=-1
    )

    predicted_index = probabilities.argmax(
        dim=-1
    ).item()

    predicted_choice = item["choices"][
        predicted_index
    ]

    is_correct = (
        predicted_choice == item["answer"]
    )

    if is_correct:
        correct += 1

    print("\nQuestion:", item["question"])
    print("Choices:", item["choices"])
    print("Probabilities:", probabilities)
    print("Predicted:", predicted_choice)
    print("Answer:", item["answer"])
    print("Correct:", is_correct)

accuracy = correct / total

print("\nAccuracy:", accuracy)
#%% [60] Test Different Number of Choices

test_state = "日本についての情報"
test_question = "通貨は何ですか？"

test_choices = [
    "円",
    "ドル",
    "ユーロ",
    "ポンド"
]

scores = score_choices(
    test_state,
    test_question,
    test_choices
)

probabilities = torch.softmax(
    scores,
    dim=-1
)

predicted_index = probabilities.argmax(
    dim=-1
).item()

predicted_choice = test_choices[
    predicted_index
]

print("Choices:")
print(test_choices)

print("\nScores:")
print(scores)

print("\nProbabilities:")
print(probabilities)

print("\nProbability sum:")
print(probabilities.sum())

print("\nPredicted choice:")
print(predicted_choice)
#%% [61] Calibration Evaluation Data

calibration_data = [
    {
        "state": "日本についての情報",
        "question": "首都はどこですか？",
        "choices": ["東京", "大阪", "京都"],
        "answer": "東京"
    },
    {
        "state": "日本についての情報",
        "question": "日本の通貨は？",
        "choices": ["円", "ドル", "ユーロ"],
        "answer": "円"
    },
    {
        "state": "日本についての情報",
        "question": "日本の国会議事堂がある都市は？",
        "choices": ["東京", "大阪", "名古屋"],
        "answer": "東京"
    },
    {
        "state": "大阪府についての情報",
        "question": "府庁所在地は？",
        "choices": ["大阪市", "京都市", "神戸市"],
        "answer": "大阪市"
    },
    {
        "state": "京都府についての情報",
        "question": "府庁所在地は？",
        "choices": ["京都市", "大阪市", "奈良市"],
        "answer": "京都市"
    },
    {
        "state": "日本についての情報",
        "question": "日本の北に位置する都道府県は？",
        "choices": ["北海道", "沖縄県", "福岡県"],
        "answer": "北海道"
    },
    {
        "state": "日本についての情報",
        "question": "日本の南に位置する県として知られるのは？",
        "choices": ["沖縄県", "北海道", "青森県"],
        "answer": "沖縄県"
    },
    {
        "state": "日本についての情報",
        "question": "富士山がまたがる都道府県は？",
        "choices": ["静岡県と山梨県", "東京都と神奈川県", "長野県と岐阜県"],
        "answer": "静岡県と山梨県"
    }
]

print("Number of calibration examples:", len(calibration_data))

for i, item in enumerate(calibration_data):
    print(f"\nExample {i + 1}")
    print("Question:", item["question"])
    print("Choices:", item["choices"])
    print("Answer:", item["answer"])
#%% [62] Get Predictions and Probabilities

evaluation_results = []

for item in calibration_data:

    scores = score_choices(
        item["state"],
        item["question"],
        item["choices"]
    )

    probabilities = torch.softmax(
        scores,
        dim=-1
    )

    predicted_index = probabilities.argmax(
        dim=-1
    ).item()

    correct_index = item["choices"].index(
        item["answer"]
    )

    evaluation_results.append({
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"],
        "correct_index": correct_index,
        "predicted_index": predicted_index,
        "probabilities": probabilities.detach().cpu()
    })

for result in evaluation_results:

    print("\nQuestion:", result["question"])
    print("Probabilities:", result["probabilities"])
    print("Predicted:", result["choices"][result["predicted_index"]])
    print("Answer:", result["answer"])
#%% [63] Calculate Accuracy

correct = 0

for result in evaluation_results:

    if result["predicted_index"] == result["correct_index"]:
        correct += 1

accuracy = correct / len(evaluation_results)

print("Correct:", correct)
print("Total:", len(evaluation_results))
print("Accuracy:", accuracy)
#%% [64] Calculate NLL

import math

nll = 0.0

for result in evaluation_results:

    probabilities = result["probabilities"]

    correct_index = result["correct_index"]

    correct_probability = probabilities[
        correct_index
    ].item()

    nll -= math.log(
        max(correct_probability, 1e-12)
    )

nll /= len(evaluation_results)

print("NLL:", nll)
#%% [65] Calculate Brier Score

brier_score = 0.0

for result in evaluation_results:

    probabilities = result["probabilities"]

    correct_index = result["correct_index"]

    target = torch.zeros_like(probabilities)

    target[correct_index] = 1.0

    squared_error = (
        probabilities - target
    ) ** 2

    brier_score += squared_error.sum().item()

brier_score /= len(evaluation_results)

print("Brier Score:", brier_score)
#%% [66] Calculate ECE

def calculate_ece(evaluation_results, num_bins=10):

    confidences = []
    correctness = []

    for result in evaluation_results:

        probabilities = result["probabilities"]

        predicted_index = result["predicted_index"]

        confidence = probabilities[
            predicted_index
        ].item()

        is_correct = (
            predicted_index == result["correct_index"]
        )

        confidences.append(confidence)
        correctness.append(float(is_correct))

    ece = 0.0

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    for i in range(num_bins):

        lower = bin_boundaries[i].item()
        upper = bin_boundaries[i + 1].item()

        if i == num_bins - 1:
            in_bin = [
                lower <= c <= upper
                for c in confidences
            ]
        else:
            in_bin = [
                lower <= c < upper
                for c in confidences
            ]

        if not any(in_bin):
            continue

        bin_confidences = [
            c for c, flag in zip(confidences, in_bin)
            if flag
        ]

        bin_correctness = [
            c for c, flag in zip(correctness, in_bin)
            if flag
        ]

        bin_confidence = sum(
            bin_confidences
        ) / len(bin_confidences)

        bin_accuracy = sum(
            bin_correctness
        ) / len(bin_correctness)

        bin_weight = (
            len(bin_confidences)
            / len(confidences)
        )

        ece += (
            abs(bin_accuracy - bin_confidence)
            * bin_weight
        )

    return ece


ece = calculate_ece(
    evaluation_results
)

print("ECE:", ece)
#%% [67] Prepare Reliability Diagram Data

def get_reliability_data(
    evaluation_results,
    num_bins=10
):

    confidences = []
    correctness = []

    for result in evaluation_results:

        probabilities = result["probabilities"]

        predicted_index = result["predicted_index"]

        confidence = probabilities[
            predicted_index
        ].item()

        is_correct = (
            predicted_index == result["correct_index"]
        )

        confidences.append(confidence)
        correctness.append(float(is_correct))

    reliability_data = []

    for i in range(num_bins):

        lower = i / num_bins
        upper = (i + 1) / num_bins

        if i == num_bins - 1:
            indices = [
                j for j, c in enumerate(confidences)
                if lower <= c <= upper
            ]
        else:
            indices = [
                j for j, c in enumerate(confidences)
                if lower <= c < upper
            ]

        if not indices:
            continue

        avg_confidence = sum(
            confidences[j]
            for j in indices
        ) / len(indices)

        accuracy = sum(
            correctness[j]
            for j in indices
        ) / len(indices)

        reliability_data.append({
            "bin": f"{lower:.1f}-{upper:.1f}",
            "confidence": avg_confidence,
            "accuracy": accuracy
        })

    return reliability_data


reliability_data = get_reliability_data(
    evaluation_results
)

for item in reliability_data:
    print(item)
#%% [68] Reliability Diagram

import matplotlib.pyplot as plt

confidences = [
    item["confidence"]
    for item in reliability_data
]

accuracies = [
    item["accuracy"]
    for item in reliability_data
]

plt.figure(figsize=(6, 6))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    confidences,
    accuracies,
    marker="o",
    label="Model"
)

plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)

plt.show()
#%% [69] Temperature Scaling

temperature = torch.tensor(
    1.0,
    device=device,
    requires_grad=True
)

temperature_optimizer = torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=100
)

print("Initial temperature:")
print(temperature.item())
#%% [70] Temperature Scaling Loss

all_logits = []
all_labels = []

for result in evaluation_results:

    probabilities = result["probabilities"]

    logits = torch.log(
        probabilities
    )

    all_logits.append(logits)

    all_labels.append(
        result["correct_index"]
    )

all_logits = torch.stack(
    all_logits
).to(device)

all_labels = torch.tensor(
    all_labels,
    device=device
)

print("Logits shape:")
print(all_logits.shape)

print("Labels shape:")
print(all_labels.shape)
#%% [71] Optimize Temperature

def temperature_closure():

    temperature_optimizer.zero_grad()

    scaled_logits = (
        all_logits / temperature
    )

    loss = criterion(
        scaled_logits,
        all_labels
    )

    loss.backward()

    return loss


temperature_optimizer.step(
    temperature_closure
)

print("Optimized temperature:")
print(temperature.item())
#%% [72] Calibrated Probabilities

with torch.no_grad():

    calibrated_logits = (
        all_logits / temperature
    )

    calibrated_probabilities = torch.softmax(
        calibrated_logits,
        dim=-1
    )

print("Calibrated probabilities:")

print(calibrated_probabilities)
#%% [73] Evaluate Calibrated NLL

calibrated_nll = criterion(
    calibrated_logits,
    all_labels
).item()

print("Original NLL:")
print(nll)

print("Calibrated NLL:")
print(calibrated_nll)
#%% [74] Evaluate Calibrated ECE

calibrated_results = []

for i, result in enumerate(evaluation_results):

    calibrated_probs = (
        calibrated_probabilities[i]
        .detach()
        .cpu()
    )

    predicted_index = calibrated_probs.argmax(
        dim=-1
    ).item()

    calibrated_results.append({
        "question": result["question"],
        "choices": result["choices"],
        "answer": result["answer"],
        "correct_index": result["correct_index"],
        "predicted_index": predicted_index,
        "probabilities": calibrated_probs
    })


calibrated_ece = calculate_ece(
    calibrated_results
)

print("Original ECE:")
print(ece)

print("Calibrated ECE:")
print(calibrated_ece)
#%% [75] Phase 4 Calibration Summary

original_accuracy = accuracy

calibrated_accuracy = sum(
    result["predicted_index"] == result["correct_index"]
    for result in calibrated_results
) / len(calibrated_results)

print("===== Phase 4 Calibration Summary =====")

print("\nAccuracy")
print("Before:", original_accuracy)
print("After :", calibrated_accuracy)

print("\nNLL")
print("Before:", nll)
print("After :", calibrated_nll)

print("\nBrier Score")
print("Before:", brier_score)

print("\nECE")
print("Before:", ece)
print("After :", calibrated_ece)

print("\nTemperature")
print(temperature.item())

# %%
