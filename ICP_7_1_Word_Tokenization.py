import nltk

f = open('output.txt').read()

wtokens = nltk.word_tokenize(f)
wtokens = [word.lower() for word in wtokens if word.isalpha()]

# Word Tokenization. Words are separated from each other.
for t in wtokens:
    print(t)
