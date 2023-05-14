import re  
import matplotlib.pyplot as plt  
from wordcloud import WordCloud, STOPWORDS  
text = open("/home/dell/DA Vrushali/Practicles/slip19.txt", "r").read()  
# Clean text  
text = re.sub(r'==.*?==+', '', text)  
text = text.replace('\n', '')  
# Define a function to plot word cloud  
def plot_cloud(wordcloud):  
    # Set figure size  
    plt.figure(figsize=(40, 30))  
    # Display image  
    plt.imshow(wordcloud)   
    # No axis details  
    plt.axis("off")  
# Generate word cloud  
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(text)
plot_cloud(wordcloud) 
plt.show()
