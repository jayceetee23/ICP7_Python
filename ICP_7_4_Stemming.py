from nltk.stem import PorterStemmer

f = open('output.txt').read()

# Displays root word by removing the ending such as ('s', 'ing', 'ed', etc.)

pStemmer = PorterStemmer()

for x in f.split():
    print(pStemmer.stem(x))