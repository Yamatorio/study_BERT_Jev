import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. 計算デバイスを選択
# ============================================================

def select_device():
    print("\n=== Device Selection ===")
    print("1: CPU")
    print("2: MPS (Apple Silicon)")
    print("3: CUDA (NVIDIA GPU)")

    while True:
        choice = input("計算デバイスを選択してください [1-3]: ").strip()

        if choice == "1":
            return torch.device("cpu")

        elif choice == "2":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                print("MPSは利用できません。別のデバイスを選択してください。")

        elif choice == "3":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("CUDAは利用できません。別のデバイスを選択してください。")

        else:
            print("1〜3を入力してください。")


# ============================================================
# 2. Model
# ============================================================

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"


def load_model(device):

    print("\nモデルを読み込んでいます...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)

    bert = bert.to(device)
    bert.eval()

    # Choiceごとのスコアを計算するHead
    choice_scorer = nn.Linear(768, 1)
    choice_scorer = choice_scorer.to(device)
    choice_scorer.eval()

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")

    return tokenizer, bert, choice_scorer


# ============================================================
# 3. Decision
# ============================================================

def decide(
    state,
    question,
    choices,
    tokenizer,
    bert,
    choice_scorer,
    device
):

    # State + Question + Choice
    texts = [
        f"{state}。{question} {choice}"
        for choice in choices
    ]

    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # BERT → Choice Score → Probability
    with torch.no_grad():

        outputs = bert(**inputs)

        # CLS representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # 各ChoiceのScore
        scores = choice_scorer(
            cls_embeddings
        ).squeeze(-1)

        # Score → Probability
        probabilities = torch.softmax(
            scores,
            dim=-1
        )

        # 最大確率のChoice
        predicted_index = probabilities.argmax().item()

    return scores, probabilities, predicted_index


# ============================================================
# 4. Interactive CLI
# ============================================================

def main():

    print("=" * 60)
    print("Mini-Jev")
    print("=" * 60)

    # Device
    device = select_device()

    # Model
    tokenizer, bert, choice_scorer = load_model(device)

    print("\n準備完了しました。")
    print("終了する場合は State に 'exit' と入力してください。")

    while True:

        print("\n" + "-" * 60)

        # State
        state = input("State: ").strip()

        if state.lower() == "exit":
            break

        if not state:
            print("Stateを入力してください。")
            continue

        # Question
        question = input("Question: ").strip()

        if not question:
            print("Questionを入力してください。")
            continue

        # Choices
        print("\nChoiceを1行ずつ入力してください。")
        print("入力終了は空行です。")

        choices = []

        while True:

            choice = input(
                f"Choice {len(choices) + 1}: "
            ).strip()

            if not choice:
                break

            choices.append(choice)

        # Choice validation
        if len(choices) < 2:
            print("Choiceは2つ以上必要です。")
            continue

        # Decision
        scores, probabilities, predicted_index = decide(
            state,
            question,
            choices,
            tokenizer,
            bert,
            choice_scorer,
            device
        )

        # Result
        print("\n" + "=" * 60)
        print("Decision Result")
        print("=" * 60)

        for i, (choice, score, probability) in enumerate(
            zip(
                choices,
                scores,
                probabilities
            )
        ):

            marker = " ← Decision" if i == predicted_index else ""

            print(
                f"{choice:20s} "
                f"score={score.item():8.4f} "
                f"prob={probability.item():.4f}"
                f"{marker}"
            )

        print("=" * 60)


# ============================================================
# 5. Entry Point
# ============================================================

if __name__ == "__main__":
    main()