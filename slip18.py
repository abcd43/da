# Import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

paragraph_text="""Hello all, Welcome to Python Programming Academy. Python
Programming Academy is a nice platform to learn new programming skills. It is
difficult to get enrolled in this Academy."""
# Word Tokenization

tokenized_text_data=sent_tokenize(paragraph_text)
tokenized_words=word_tokenize(paragraph_text)
print("Tokenized Sentences : \n", tokenized_text_data, "\n")
print("Tokenized Words : \n",tokenized_words, "\n")
frequency_distribution=FreqDist(tokenized_words)
print(frequency_distribution)
print(frequency_distribution.most_common(2))
frequency_distribution.plot(32,cumulative=False)
plt.show()
