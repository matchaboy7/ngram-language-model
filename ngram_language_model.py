import string
import re
import random
import math
from collections import defaultdict

############################################################
# Section 1: Ngram Models
############################################################


def tokenize(text):
    """
    This method splits input text string into a list of tokens.
    :param text: a string of text.
    :return: a list of tokens derived from that text.
    The token should be a contiguous sequence of non-whitespace
    characters, with the exception that any punctuation mark
    should be treated as an individual token.
    """
    # Get all punctuation characters
    punctuation = string.punctuation

    # Escape all punctuation characters
    # as literal characters in the regex pattern
    # e.g. "[" becomes "\["
    escaped_punctuation = re.escape(punctuation)

    # Process raw formatted string:
    # match one or more consecutive letters/digits/underscore
    # OR match any single punctuation.
    pattern = rf"[\w]+|[{escaped_punctuation}]"

    # Tokenizing: use regular expression for matching
    tokens = re.findall(pattern, text)
    return tokens


def ngrams(n, tokens):
    """
    This method produces a list of all n-grams of
    the specified size from the input token list.
    :param n: Each n-gram should consist of a 2-element
    tuple (context, token), where the context is itself
    an (n − 1)-element tuple comprised of the n − 1 words
    preceding the current token. Assume that n ≥ 1.
    :return: a sentence that is padded with n − 1 "<START>"
    tokens at the beginning and a single "< END>"
    token at the end. If n = 1, all contexts should
    be empty tuples.
    """
    # create padded token list
    padded_tokens = ["<START>"] * (n - 1) + tokens + ["<END>"]
    ngrams_list = []  # store the n-grams list

    for i in range(n - 1, len(padded_tokens)):
        # tuple [i-(n-1):i], (n-1) words before the current token:
        context = tuple(padded_tokens[i - n + 1:i])
        # the current token:
        token = padded_tokens[i]
        ngrams_list.append((context, token))
    return ngrams_list


class NgramModel(object):

    def __init__(self, n):
        """
        This method stores the order n of the model
        and initializes dictionaries to store
        n-gram counts and context counts.
        """
        self.n = n
        # initializes two dictionaries:
        # NOTE:
        # defaultdict(int) initializes missing keys as 0.
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    def update(self, sentence):
        """
        This method computes the n-grams for the input
        sentence and updates the internal counts.
        :param sentence: return from ngrams(), it is a sentence
        that is padded with n − 1 "<START>" tokens at the
        beginning and a single "< END>" token at the end.

        1. ngram_counts: to store the count of each (context, token) pair.
        e.g.
        ngram_counts = {
            (('<START>',), 'a'): 1,
            (('a',), 'b'): 2,
            (('b',), 'a'): 1,
            (('b',), '<END>'): 1
        }
        2. context_counts: to store the total count of each context.
        e.g.
        context_counts = {
            ('<START>',): 1,
            ('a',): 2,
            ('b',): 2
        }
        """
        # Tokenize input sentence:
        tokens = tokenize(sentence)
        # generate n-grams list:
        ngrams_list = ngrams(self.n, tokens)

        # For each (context, token) pair in the n-grams list,
        # update the n-gram and context counts:
        for context, token in ngrams_list:
            self.ngram_counts[(context, token)] += 1
            self.context_counts[context] += 1

    def prob(self, context, token):
        """
        This method accepts an (n − 1)-tuple representing
        a context and a token
        :return: returns the probability of that token occurring,
        given the preceding context.

        calculate the probability of a token with given context:
            P(token|context) = Count(context, token) / Count(context)

        In an n-gram model, P(token∣context) represents the
        conditional probability of predicting "token" given
        the preceding "context". This probability is estimated
        based on the frequencies of (context, token) pairs and
        context frequencies in the given "text".
        """
        ngram_count = self.ngram_counts[(context, token)]
        context_count = self.context_counts[context]

        # conditional probability P(token|context):
        if context_count > 0:
            return ngram_count / context_count
        else:
            return 0.0

    def random_token(self, context):
        """
        :return: a random token according to the
        probability distribution determined by the given context.
        This method returns the first token such that the
        cumulative probability of the given "context"
        ≥ random threshold r.
        """
        # create empty list to store tokens
        tokens = []
        # iterate each (context, token) pair in ngram_counts
        for (c, token) in self.ngram_counts:
            # if current context c matches given context:
            if c == context:
                # add c's token to tokens list:
                tokens.append(token)
        # sort in natural lexicographic ordering
        sorted_tokens = sorted(tokens)

        # Initialize cumulative probability
        context_cumulative_prob = 0.0
        # generate a random number r, range in [0, 1)
        r = random.random()

        for t in sorted_tokens:
            # accumulate prob of each token in given context:
            context_cumulative_prob += self.prob(context, t)
            # return the first token exceeds r
            if context_cumulative_prob > r:
                return t

        # edge cases:
        return sorted_tokens[-1]

    def random_text(self, token_count):
        """
        :param token_count: The number of tokens
        to be generated.
        :return: a string of space-separated tokens
        (length of token_count) chosen at random using
        the random_token(self, context) method.
        """
        # Step 1: initialize context
        if self.n == 1:  # context is empty tuple for n = 1
            context = ()
        else:
            # initialize starting context be (n−1)-tuple:
            context = ("<START>",) * (self.n - 1)

        # Step 2: generate tokens
        tokens = []  # store tokens for method return

        # iterate "token_count" times to generate
        # token_count" number of tokens:
        for _ in range(token_count):
            # get an addition token based on current context:
            token = self.random_token(context)
            tokens.append(token)

            # Step 3: update context:
            # reset the context to the starting context
            # if "<END>" is encountered:
            if token == "<END>":
                if self.n == 1:
                    context = ()
                else:
                    context = ("<START>",) * (self.n - 1)
            else:
                # shifting and adding the new token
                if self.n > 1:
                    context = context[1:] + (token,)

        return ' '.join(tokens)

    def perplexity(self, sentence):
        """
        This method computes the n-grams for the input sentence
        and returns their perplexity under the current model.
        P(w1, w2, ..., wm): the probability of generating the
        entire sequence of m tokens (w1, w2, ..., wm, wm+1)
        under the given model.

        log_prob_sum: ∑(log P(w_i | w_{i-1}))
        """
        # tokenize the input sentence
        tokens = tokenize(sentence)

        # to generate n-grams list
        ngrams_list = ngrams(self.n, tokens)
        log_prob_sum = 0.0
        N = len(ngrams_list)  # number of tokens = m+1

        for context, token in ngrams_list:
            prob = self.prob(context, token)

            # poor model fit: infinite perplexity
            if prob == 0:
                return float('inf')

            log_prob_sum += math.log(prob)

        avg_log_prob = log_prob_sum / N

        return math.exp(-avg_log_prob)


def create_ngram_model(n, path):
    """
    This method loads the text at the given path
    and creates an n-gram model from the resulting data.
    Each line in file is treated as a separate sentence.
    :param n: the n-gram model (e.g. 1 as unigram, 2 as bigram).
    :param path: The path to the text file containing the data.
    :return: an instance of NgramModel based on the training
    from the text located at "path".
    """
    # Initialize NgramModel
    model = NgramModel(n)

    # Open file and read all the lines
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # strip whitespace and skip empty lines:
            sentence = line.strip()
            if sentence:
                # Tokenize "sentence", and generate n-grams
                # list, and update counters:
                model.update(sentence)
    return model


def main():
    # Test tokenize()
    print("tokenize Test 1")
    text1 = " This is an example. "
    expected_output1 = [
        'This', 'is', 'an',
        'example', '.'
    ]
    output1 = tokenize(text1)
    print("tokenize Test 1 Passed"
          if output1 == expected_output1
          else "Test 1 Failed")
    print("Output:", output1)
    print("tokenize Test 2")
    text2 = "'Medium-rare,' she said."
    expected_output2 = [
        "'", 'Medium', '-', 'rare',
        ',', "'", 'she', 'said', '.'
    ]
    output2 = tokenize(text2)
    print("tokenize Test 2 Passed"
          if output2 == expected_output2
          else "Test 2 Failed")
    print("Output:", output2)
    print()

    # Test ngrams()
    print("N-grams Test 1")
    n1, tokens1 = 1, ["a", "b", "c"]
    expected_output_ngram1 = [
        ((), 'a'),
        ((), 'b'),
        ((), 'c'),
        ((), '<END>')
    ]
    output_ngram1 = ngrams(n1, tokens1)
    print("N-grams Test 1 Passed"
          if output_ngram1 == expected_output_ngram1
          else "N-grams Test 1 Failed")
    print("Output:", output_ngram1)
    print("N-grams Test 2")
    n2, tokens2 = 2, ["a", "b", "c"]
    expected_output_ngram2 = [
        (('<START>',), 'a'),
        (('a',), 'b'),
        (('b',), 'c'),
        (('c',), '<END>')
    ]
    output_ngram2 = ngrams(n2, tokens2)
    print("N-grams Test 2 Passed"
          if output_ngram2 == expected_output_ngram2
          else "N-grams Test 2 Failed")
    print("Output:", output_ngram2)
    print("N-grams Test 3")
    n3, tokens3 = 3, ["a", "b", "c"]
    expected_output_ngram3 = [
        (('<START>', '<START>'), 'a'),
        (('<START>', 'a'), 'b'),
        (('a', 'b'), 'c'),
        (('b', 'c'), '<END>')
    ]
    output_ngram3 = ngrams(n3, tokens3)
    print("N-grams Test 3 Passed"
          if output_ngram3 == expected_output_ngram3
          else "N-grams Test 3 Failed")
    print("Output:", output_ngram3)
    print()

    # Test random_token()
    print("random_token Test 1")
    m = NgramModel(1)
    m.update("a b c d")
    m.update("a b a b")
    random.seed(1)
    tokens_case1 = [m.random_token(())
                    for _ in range(25)]
    print(tokens_case1)
    expected_output = ['<END>', 'c', 'b', 'a', 'a',
                       'a', 'b', 'b', '<END>', '<END>',
                       'c', 'a', 'b', '<END>', 'a', 'b',
                       'a', 'd', 'd', '<END>', '<END>',
                       'b', 'd', 'a', 'a']
    if tokens_case1 == expected_output:
        print("random_token Test 1 passes.")
    print("random_token Test 2")
    m = NgramModel(2)
    m.update("a b c d")
    m.update("a b a b")
    random.seed(2)
    tokens_case2_start = [m.random_token(("<START>",))
                          for _ in range(6)]
    print(tokens_case2_start)
    expected_output = ['a', 'a', 'a', 'a', 'a', 'a']
    if tokens_case2_start == expected_output:
        print("random_token Test 2 tokens_case2_start passes.")
    tokens_case2_b = [m.random_token(("b",))
                      for _ in range(6)]
    print(tokens_case2_b)
    expected_output = ['c', '<END>', 'a', 'a', 'a', '<END>']
    if tokens_case2_b == expected_output:
        print("random_token Test 2 tokens_case2_b passes.")
    print()

    # Test random_text()
    # Test case 1
    print("random_text Test case 1:")
    m = NgramModel(1)
    m.update("a b c d")
    m.update("a b a b")
    random.seed(1)
    expected = '<END> c b a a a b b <END> <END> c a b'
    if m.random_text(13) == expected:
        print("random_text() Test case 1 passes")
    # Test case 2:
    print("random_text Test case 2:")
    m = NgramModel(2)
    m.update("a b c d")
    m.update("a b a b")
    random.seed(2)
    expected = 'a b <END> a b c d <END> a b a b a b c'
    if m.random_text(15) == expected:
        print("random_text() Test case 2 passes")
    print()


if __name__ == "__main__":
    main()
