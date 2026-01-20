from collections import Counter, deque
from functools import lru_cache
import json
import re


class BPETokenizer:
    def __init__(self):
        # Maps token_id -> token string
        self.vocab = {}
        # Maps token string -> token_id
        self.inverse_vocab = {}
        # BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}
        # Rank dictionary for merges: {(string_A, string_B): rank}
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special=None):
        """
        Train the BPE tokenizer (GPT-2 style):
        - Inserts Ġ before words to mark spaces
        - Learns merges only within words (never across Ġ)
        """
        # --- Step 1: Preprocess into GPT-2 style tokens ---
        processed_text = []
        for i, ch in enumerate(text):
            if ch == " " and i != 0:
                processed_text.append("Ġ")  # mark space
            if ch != " ":
                processed_text.append(ch)
        processed_text = "".join(processed_text)

        # --- Step 2: Initialize base vocab ---
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(c for c in sorted(set(processed_text)) if c not in unique_chars)
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: ch for i, ch in enumerate(unique_chars)}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}

        # --- Step 3: Add special tokens if provided ---
        if allowed_special:
            for tok in allowed_special:
                if tok not in self.inverse_vocab:
                    nid = len(self.vocab)
                    self.vocab[nid] = tok
                    self.inverse_vocab[tok] = nid

        # --- Step 4: Tokenize into IDs (character level) ---
        token_ids = [self.inverse_vocab[ch] for ch in processed_text]

        # --- Step 5: Iterative BPE merges ---
        while len(self.vocab) < vocab_size:
            pair_id = self.find_freq_pair(token_ids, self.vocab, mode="most")
            if pair_id is None:
                break

            new_id = len(self.vocab)
            self.bpe_merges[pair_id] = new_id

            p0, p1 = pair_id
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

            rank = len(self.bpe_ranks)
            self.bpe_ranks[(self.vocab[p0], self.vocab[p1])] = rank

            token_ids = self.replace_pair(token_ids, pair_id, new_id)

    def encode(self, text, allowed_special=None):
        """
        Encode text into token IDs.
        - Preserves special tokens if allowed
        - Inserts Ġ before words (GPT-2 convention)
        - Applies BPE merges to each word
        """
        token_ids = []

        # --- Handle special tokens ---
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )

            last_index = 0
            for match in re.finditer(special_pattern, text):
                # Encode text before special token
                prefix = text[last_index:match.start()]
                if prefix.strip():
                    token_ids.extend(self.encode(prefix, allowed_special=None))

                # Add special token directly
                special_token = match.group(0).strip()
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocab.")
                last_index = match.end()

            text = text[last_index:]

        # --- Normal text encoding (GPT-2 style) ---
        if text.strip():
            words = []
            for i, w in enumerate(text.split()):
                if i == 0:
                    words.append(w)
                else:
                    words.append("Ġ" + w)

            for word in words:
                if word in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[word])
                else:
                    token_ids.extend(self.tokenize_with_bpe(word))

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Apply BPE merges to a single token (e.g., "The" or "Ġcat").
        """
        symbols = list(token)

        # Iteratively merge pairs using learned ranks
        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        return [self.inverse_vocab[sym] for sym in symbols if sym in self.inverse_vocab]

    def decode(self, token_ids, skip_special=None):
        """
        Decode a list of token IDs back into text.
        - Ġ becomes a space
        - Skips tokens in skip_special set (e.g., {"<eos>"})
        """
        if skip_special is None:
            skip_special = set()

        decoded_string = ""
        for tid in token_ids:
            if tid not in self.vocab:
                raise ValueError(f"Token ID {tid} not found in vocab.")
            token = self.vocab[tid]

            # --- Skip specified special tokens ---
            if token in skip_special:
                continue

            if token == "Ġ":
                decoded_string += " "
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Save vocab + merges to disk as JSON."""
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Load vocab + merges from disk."""
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, vocab, mode="most"):
        """
        Find the most/least frequent adjacent pair,
        ignoring pairs that cross Ġ (word boundary).
        """
        pairs = Counter()
        for a, b in zip(token_ids, token_ids[1:]):
            sa, sb = vocab[a], vocab[b]
            # 🚫 Block merges across space marker Ġ
            if sa == "Ġ" or sb == "Ġ":
                continue
            pairs[(a, b)] += 1

        if not pairs:
            return None
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        """Replace all occurrences of a pair with the merged token ID."""
        dq = deque(token_ids)
        replaced = []
        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)
        return replaced


# --- Example usage ---
if __name__ == "__main__":
    corpus = "the cat in the hat"
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=300, allowed_special={"<eos>"})

    test = "the cat <eos>"
    toks = bpe.encode(test, allowed_special={"<eos>"})
    print("Input:", test)
    print("Tokens:", toks)
    print("Decoded (skip <eos>):", bpe.decode(toks, skip_special={"<eos>"}))
