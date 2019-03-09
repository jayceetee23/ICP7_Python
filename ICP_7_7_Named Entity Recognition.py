from nltk import wordpunct_tokenize, pos_tag, ne_chunk

f = open('output.txt').read()

#  Classify certain elements in text into pre-defined categories (Person, organization, expressions of times, etc.)
print(ne_chunk(pos_tag(wordpunct_tokenize(f))))

