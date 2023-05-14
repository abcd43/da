import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
text="""so keep working.Keep striving never give up fall down seven times, get eight.Ease greater threat to progress than hardship. Ease greater threat to progress than hardship.So, keep moving, keep growing ,and keep learning.See you at work.Process the text to remove any special character and digits.Generate the summary using extractive summarization process"""
text1 = re.sub(r'\[[0-9]*\]', ' ', text)
text2= re.sub(r'\s+', ' ', text1)
text3= re.sub('[^0-9]', ' ', text)
formatted_text = re.sub('[^a-zA-Z]', ' ', text)
print(text1)
print(text2)
print(text3)
print(formatted_text)

