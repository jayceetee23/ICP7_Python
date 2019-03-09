import nltk

f = open('output.txt').read()

stokens = nltk.sent_tokenize(f)

# Sentence Tokenization. Sentences are separated from each other.
for s in stokens:
    print(s)
