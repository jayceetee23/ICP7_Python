from nltk.stem import WordNetLemmatizer

f = open('output.txt').read()


# Lemmatization obtains word by determining the part of speech of the word and applies different normalization rules
# for each part of speech

lemmatizer = WordNetLemmatizer()

for x in f.split():
    print(lemmatizer.lemmatize(x, 'v'))
