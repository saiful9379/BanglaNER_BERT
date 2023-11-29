"""

Basic tokenization tokenize sentence using white spaces, punctuation mark
Code shamelessly copied from BERT tokenization 
To check Original Code: https://github.com/google-research/bert/blob/master/tokenization.py

"""
import json
import six
import unicodedata
from typing import Any, List, Dict, Set


class BasicTokenizer:
  """
  Runs basic tokenization punctuation, splitting and lower casing
  """
  def whitespace_tokenize(self, text):
    """
    Runs basic whitespace cleaning and splitting on a piece of text
    """
    text = text.strip()
    # print("Text: ", text)
    if not text:
        return []
    tokens = text.split()
    # print("tokens : ", tokens)
    return tokens
  
  def convert_to_unicode(self, text:str):
    """
    Converts text to Unicode assuming utf-8 .
    
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

  def _run_strip_accents(self, text):
    """
    Strips accents from a piece of text.
    
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)

    return "".join(output)
  
  def _is_punctuation(self, char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if self._is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]
  
  def tokenize(self, text):
    """
    Tokenizes a piece of text.
    
    """
    text = self.convert_to_unicode(text)
    orig_tokens = self.whitespace_tokenize(text)
    # print("original tokens: ", orig_tokens)
    split_tokens = []
    for token in orig_tokens:
        # if self.do_lower_case:
        #   token = token.lower()
        #   token = self._run_strip_accents(token)
        split_tokens.extend(self._run_split_on_punc(token))

    # print("split tokens: ", split_tokens)
    output_tokens = self.whitespace_tokenize(" ".join(split_tokens))
    return output_tokens



class DataProcessing:
  
  def __init__(self, file_path):
    self.file_path = file_path
  
  def read_jsonl_file(self):
    """
    read jsonl file
    """
    with open(self.file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data
  
  def process_jsonl_file(self):
    """
    
    Check data processing
    """
    data = self.read_jsonl_file()
    bst = BasicTokenizer()
    match_line = 0
    data_list = []
    for line in data:
        text, labels = line[0], line[1]
        tokens = bst.tokenize(text)
        # print(tokens)
        if len(tokens) != len(labels):
                print(tokens, 'length', len(tokens))
                print(labels, 'lenght', len(labels))
                print('-'*20)
        else:
            match_line += 1
            data_list.append([text, tokens, labels])
    print(f"Match {match_line}, Total {len(data)}, Not Match: {len(data)-match_line}")

    return data_list

if __name__ == '__main__':

    file_path = 'data/main.jsonl'
    dp = DataProcessing(file_path)
    dp.process_jsonl_file()