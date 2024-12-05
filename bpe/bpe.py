import regex
from collections import Counter

class BytePairTokenizer:
  def __init__(self,
               target_vocab_size,
               pattern=r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    self.target_vocab_size = target_vocab_size
    self.merges = []
    self.pattern = pattern
  
  def _compute_word_frequencies(self, corpus):
    """Compute word frequencies from input corpus."""
    return Counter([word for word in regex.findall(self.pattern, " ".join(corpus))])
  
  def _initialize_tokens(self, word_frequency):
    """Initialize tokens as single characters."""
    return {word:list(word) for word in word_frequency}
  
  def _compute_pair_frequencies(self, tokens, word_frequency):
    """Compute frequencies of adjacent token pairs."""
    pair_freq = Counter()
    for word, freq in word_frequency.items():
      token = tokens[word]
      for i in range(len(token) - 1):
        pair_freq[tuple(token[i:i+2])] += freq
    return pair_freq
  
  def _find_best_pair(self, pair_freq):
    """Find most frequent pair in our tokenized corpus"""
    return pair_freq.most_common(1)[0][0] if pair_freq else None

  def _merge_tokens(self, tokens, pair):
    """Replace occurences of a pair of tokens with the merged form."""
    merged_tokens = {}
    merge_str = ''.join(pair)
    for word, token in tokens.items():
      new_token = []
      i = 0
      while i < len(token):
        if i < len(token) - 1 and tuple(token[i:i+2]) == pair:
          new_token.append(merge_str)
          i += 2
        else:
          new_token.append(token[i])
          i += 1
      merged_tokens[word] = new_token
    return merged_tokens

  def fit(self, corpus):
    word_frequency = self._compute_word_frequencies(corpus)
    tokens = self._initialize_tokens(word_frequency)

    while len(self.merges) + len(set("".join(word_frequency.keys()))) < self.target_vocab_size:
      pair_frequency = self._compute_pair_frequencies(tokens, word_frequency)
      best_pair = self._find_best_pair(pair_frequency)
      if not best_pair:
        break
      self.merges.append(best_pair)
      tokens = self._merge_tokens(tokens, best_pair)
  
  def transform(self, corpus):
    """Iteratively merge learnt pairs of tokens in a corpus."""
    merge_dict = {m:m[0]+m[1] for m in self.merges}

    tokenized_corpus = []
    for document in corpus:

        document = list(document)

        for merge_pair, merged_token in merge_dict.items():
          i = 0
# Todo I can speed this up by a lot! need to check sets of tokens against merge pairs so we can do depth first merging!
          while i < len(document) - 1:
            if tuple(document[i:i + len(merge_pair)]) == merge_pair:
              document = document[:i] + [merged_token] + document[i + len(merge_pair):]
            i+=1
        
        tokenized_corpus.append(document)
    return tokenized_corpus
  
class Vectorizer:
  def __init__(self):
    self.vocabulary = {}
    self.inverse_vocabulary = {}
  
  def fit(self, tokenized_corpus):
    tokens = [tok for doc in tokenized_corpus for tok in doc]
    token_frequency = Counter(tokens).most_common()

    integer = 1

    for token, _ in token_frequency:
      self.vocabulary[integer] = token
      self.inverse_vocabulary[token] = integer
      integer += 1
  
  def transform(self, tokens):
    vectorized_corpus = []

    for doc in tokens:
      vectorized_document = []

      for tok in doc:
        vectorized_document.append(self.inverse_vocabulary[tok])

      vectorized_corpus.append(vectorized_document)

    return vectorized_corpus
  
  def fit_transform(self, tokens):
    self.fit(tokens)
    return self.transform(tokens)
  
  def inverse_transform(self, tokens):
    tokenized_corpus = []
    for doc in tokens:
      tokenized_document = []
      for tok in doc:
        tokenized_document.append(self.vocabulary[tok])
      tokenized_corpus.append(tokenized_document)
    return tokenized_corpus