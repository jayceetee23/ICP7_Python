import nltk

f = open('output.txt').read()

wtokens = nltk.word_tokenize(f)
wtokens = [word.lower() for word in wtokens if word.isalpha()]

# Applies label to word based on part of speech (noun, verb, adjective, etc.).
for x in wtokens:
    print(nltk.pos_tag(wtokens))
