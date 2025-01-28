import regex
from collections import Counter

PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^ \s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def list_to_dict(string_list, start_value=0):
  result_dict = {}
  for element in string_list:
    result_dict[element] = start_value
    start_value += 1
  return result_dict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.merged_token = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, merged_token):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.merged_token = merged_token

class BytePairTokenizer:
  def __init__(self,
               target_vocab_size):
    self.target_vocab_size = target_vocab_size
    self.merges = []
    self.pattern = PATTERN
  
  def _compute_word_frequencies(self, corpus):
    """Compute word frequencies from input corpus."""
    return Counter([word for word in regex.findall(self.pattern, "".join(corpus))])
  
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
  
  def _build_trie(self, merges):
    trie = Trie()
    for merge_pair, merged_token in merges.items():
        trie.insert(''.join(merge_pair), merged_token)
    return trie
  
  def transform(self, corpus):
    """
    Efficiently merges pairs of tokens in a corpus using a Trie.

    Args:
        corpus: An iterable of strings (the corpus).

    Returns:
        Merged corpus.
    """
    # Build trie from 
    merge_dict = {m: m[0] + m[1] for m in self.merges}
    trie = self._build_trie(merge_dict)

    result = []
    for document in corpus:
      i = 0
      output = []
      last_append = 0
      while i < len(document): # iterate over every char in doc
        node = trie.root
        j = i # start iterating again from that char
        last_match = None
        while j < len(document) and document[j] in node.children: # if we're in doc and char is a node
          node = node.children[document[j]] # go to that node
          if node.is_end_of_word:
            last_match = (j + 1, node.merged_token) # store the end position and the merged token
          j += 1 # keep going through trie

        if last_match: # once we left trie because char isn't a node
          end_pos, merged_token = last_match
          output.append(document[last_append:i])
          output.append(merged_token)
          i = end_pos
          last_append = i
        else:
          i += 1

      output.append(document[last_append:])
      # output.append(list("".join(document[last_append:])))
      result.append(output)
    return result

from collections import defaultdict

class Vectorizer:
  def __init__(self, vocab_size, vocabulary={}):
    self.vocab_size= vocab_size
    self.vocabulary = vocabulary
    self.inverse_vocabulary = defaultdict(int)
  
  def fit(self, tokenized_corpus):
    tokens = [tok for doc in tokenized_corpus for tok in doc]
    token_frequency = Counter(tokens).most_common(self.vocab_size - 1)
    self.vocabulary[0] = '<UNK>'
    self.inverse_vocabulary['<UNK>'] = 0

    integer = 1
    for token, _ in token_frequency:
      self.vocabulary[integer] = token
      self.inverse_vocabulary[token] = integer
      integer += 1
  
  def transform(self, tokens):
    vectorized_corpus = [
        [self.inverse_vocabulary.get(tok, 0) for tok in doc]
        for doc in tokens
    ]
    return vectorized_corpus
  
  def fit_transform(self, tokens):
    self.fit(tokens)
    return self.transform(tokens)
  
  def inverse_transform(self, tokens):
    tokenized_corpus = []
    for doc in tokens:
      tokenized_document = []
      for tok in doc:
        tokenized_document.append(self.vocabulary.get(tok, '<UNK>'))
      tokenized_corpus.append(tokenized_document)
    return tokenized_corpus