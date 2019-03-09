from nltk import ngrams

f = open('output.txt').read()

n = 3

# Splits the text
tri = ngrams(f.split(), n)

for x in tri:
    print(x)
