# BanglaNER_BERT
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U7DfMarUE61fFyzdWldpzWO7rm5EQRmQ)




# Dataset

Dataset [link](https://www.kaggle.com/datasets/saifulislam79/banglaner-dataset/data)


# Model Training

Check the training into colab

# Inference
Inference Pipeline into [huggingface](https://huggingface.co/saiful9379/BanglaNER_BERT)

```py
import torch 
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 512
labels_to_ids = {'O': 0, 'B-PER': 1, 'L-PER': 2, 'I-PER': 3, 'U-PER': 4}

ids_to_labels = {value:key for key, value in labels_to_ids.items()}

tokenizer = BertTokenizerFast.from_pretrained('saiful9379/BanglaNER_BERT')
model = BertForTokenClassification.from_pretrained('saiful9379/BanglaNER_BERT', num_labels=len(labels_to_ids))
model.to(device)

text_list = [
    ": আল স্কিনিয়ার গিটার, পিয়ানো, ভোকাল, মগ সিনথেসাইজার",
    "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম",
    "রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম",
    "সাইফুল ইসলাম ন্যাচারাল ল্যাঙ্গুয়েজে প্রসেসিং খুব বেশি ভালো পারে না । তাই সে বেশি বেশি স্টাডি করতেছে।",
    "ডিপিডিসির স্পেশাল টাস্কফোর্সের প্রধান মুনীর চৌধুরী জানান",
    "তিনি মোহাম্মদ বাকির আল-সদর এর ছাত্র ছিলেন।",
    "লিশ ট্র্যাক তৈরির সময় বেশ কয়েকজন শিল্পীর দ্বারা অনুপ্রাণিত হওয়ার কথা স্মরণ করেন, বিশেষ করে ফ্রাঙ্ক সিনাত্রা ।",
]

def inference(sentence):
  inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

  # move to gpu
  ids = inputs["input_ids"].to(device)
  mask = inputs["attention_mask"].to(device)
  # forward pass
  outputs = model(ids, mask)
  logits = outputs[0]

  active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
  flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

  tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
  token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
  wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

  word_level_predictions = []
  for pair in wp_preds:
    if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
      # skip prediction
      continue
    else:
      word_level_predictions.append(pair[1])

  str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
  print(str_rep)
  print(wp_preds)
  print(word_level_predictions)
  print("="*30)

```


# Evaluation

# Reference
