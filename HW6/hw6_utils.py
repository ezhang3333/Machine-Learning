import torch
import numpy as np
import torch.utils.data
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification

''' Start GPT-2 Finetuning Helpers '''

# --------------------------------------------------------------------
# Do Not modify the following Constants
# --------------------------------------------------------------------
MODEL_NAME = "gpt2"
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
YES_ID, NO_ID = TOKENIZER.encode(" yes")[0], TOKENIZER.encode(" no")[0]

MAX_LEN = 200  # maximum total length (prompt + answer + pads)


def load_classification_data(path="classification_train.txt"):
    """
    Read a TSV file in the form:
        <label>\t<review text>\n
    and return a Hugging Face `Dataset` with columns
        • "label"   (int 0/1)
        • "content" (str)

    Parameters
    ----------
    path : str

    Returns
    -------
    datasets.Dataset
    """
    records = []
    with open(path, encoding="utf‑8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad line {line_no} in {path!r}: {line!r}")
            label_str, text = parts
            records.append({"label": int(label_str), "content": text})
    return Dataset.from_list(records)


def prompt_zero(text):
    """
    Construct the prompt used for zero‑shot sentiment prediction.

    GPT2 is asked to generate the next token, which should be either “ yes” or “ no”.

    Parameters
    ----------
    text : str
        The raw review text.

    Returns
    -------
    str
        The full prompt string ready to be fed to the language model.
    """
    return (
        f"Review: {text}\n"
        "Is the sentiment of the review positive?\n"
        "Answer yes or no: "
    )


def eval_seq2seq(ds, lm):
    """
    Evaluate a GPT‑2 model on sentiment classification (zero‑shot seq2seq).

    Parameters
    ----------
    ds : Dataset
        Hugging Face dataset with fields `"content"` and `"label"`.
    lm : GPT2LMHeadModel
        A pretrained GPT‑2 model.

    Returns
    -------
    Tuple[float, float]
        (average NLL over the dataset, classification accuracy).
    """
    lm.eval()
    dev = next(lm.parameters()).device
    total_nll, correct = 0.0, 0

    for ex in tqdm(ds, desc="seq2seq‑eval", leave=False):
        prompt = prompt_zero(ex["content"])
        inputs = TOKENIZER(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = lm(**inputs)
            logits = outputs.logits[0, -1]
        probs = torch.softmax(logits[[YES_ID, NO_ID]], dim=0).cpu().numpy()
        p_yes, p_no = probs

        true_label = "positive" if ex["label"] == 1 else "negative"
        predicted_label = "positive" if p_yes > p_no else "negative"
        p_true = p_yes if true_label == "positive" else p_no

        total_nll -= math.log(max(p_true, 1e-12))
        correct += int(predicted_label == true_label)

    avg_nll = total_nll / len(ds)
    accuracy = correct / len(ds)
    return avg_nll, accuracy


def eval_clf(ds, model, tokenize_clf):
    """
    Evaluate a GPT‑2 classification head on a sentiment dataset.

    Parameters
    ----------
    ds : Dataset
        A Dataset containing "content" and "label" columns.

    model : GPT2ForSequenceClassification
        The fine‑tuned model, already on DEVICE.

    tokenize_clf : callable
        YOUR implemented function to tokenize the dataset.

    Returns
    -------
    Tuple[float, float]
        (average cross‑entropy loss, accuracy)
    """
    model.eval()
    dev = next(model.parameters()).device

    tok_ds = ds.map(
        tokenize_clf,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
    )

    loader = torch.utils.data.DataLoader(
        tok_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(TOKENIZER),
    )

    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="clf‑eval", leave=False):
            labels = batch["labels"].to(dev)
            outputs = model(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev)
            )
            total_loss += ce(outputs.logits, labels).item()
            correct += (outputs.logits.argmax(-1) == labels).sum().item()

    avg_loss = total_loss / len(ds)
    accuracy = correct / len(ds)
    return avg_loss, accuracy


def eval_model(model, tokenize_clf=None):
    """
    Evaluate a sentiment‑analysis model on the training set to examine whether the training is implemented correctly.
    In the gradescope, we will test the model on the heldout test set.

    This function loads the test set and
    dispatches to the correct evaluation routine based on the model type:

      • If `model` is a GPT2LMHeadModel (seq2seq), it calls `eval_seq2seq`.
      • If `model` is a GPT2ForSequenceClassification, it calls `eval_clf`.

    Parameters
    ----------
    model:
        A fine‑tuned or pretrained model. Must be either
        GPT2LMHeadModel or GPT2ForSequenceClassification.
    tokenize_clf : callable, optional.
        YOUR implemented function to tokenize the dataset for classification.

    Returns
    -------
    Tuple[float, float]
        A pair (loss, accuracy) on the test set.
    """
    # load the test data
    test_set = load_classification_data("classification_train.txt")

    # seq2seq model evaluation
    if isinstance(model, GPT2LMHeadModel):
        return eval_seq2seq(test_set, model)

    # classification head evaluation
    if isinstance(model, GPT2ForSequenceClassification):
        assert tokenize_clf is not None, "tokenize_clf must be provided for classification head"
        return eval_clf(test_set, model, tokenize_clf)

    raise ValueError(
        f"Unsupported model type: {model.__class__.__name__}. "
        "Expected GPT2LMHeadModel or GPT2ForSequenceClassification."
    )



''' End GPT-2 Finetuning Helpers '''
