    Python 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license()" for more information.
    /* The following is the source code for the project. If you are interested in what some of the code means, I have put occasional explainers for large sections of code so that the main idea is able to be derived*/

    import matplotlib
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn import metrics
    from collections import Counter
    from sklearn.metrics import classification_report
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # We need to load the Data
    disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")
    
    # This is the filter for Profanity
    !pip install better_profanity
    
    from better_profanity import profanity
    
    disaster_tweets['contains_profanity'] = disaster_tweets['text'].apply(lambda x: profanity.contains_profanity(x))
    disaster_clean = disaster_tweets[disaster_tweets['contains_profanity'] == False]
    disaster_tweets = disaster_clean
    disaster_tweets.head()
    
    tweet_set = disaster_tweets['text']
    tweet_labels = disaster_tweets['category']
    X_train, X_test, y_train, y_test = train_test_split(tweet_set, tweet_labels, test_size=0.2, random_state=1)
    
    # This is to generate labels for our plot
    tweet_categories = list(set(tweet_labels))
    
    # This is to generate counts for each tweet type
    category_counts = [np.sum(disaster_tweets["category"] == i) for i in tweet_categories]
    
    # This is to generate a bar plot for our tweet labels that has different colors
    [plt.bar(x = tweet_categories[i], height = category_counts[i] ) for i in range(len(tweet_categories))]
    
    # This is to make the plot interpretable with x and y labels + title
    plt.xlabel('TWEET CATEGORY')
    plt.ylabel('N OBSERVSATIONS')
    plt.title('A distribution of tweet categories in our data set', y=1.05);
    print(Counter(tweet_labels))
    
    def classify_rb(tweet):
    
      tweet = str(tweet).lower() 
    
    if "medicine" in tweet or "first aid" in tweet:
      return "Medical"
    elif "power" in tweet or "battery" in tweet:
      return "Energy"
    elif "water" in tweet or "bottled" in tweet:
      return "Water"
    elif "food" in tweet or "perishable" in tweet or "canned" in tweet:
      return "Food"
    else:
      return "None"
    
    # Rule Classifier on Predictions
    
    def show_pred(y_test,y_pred):
      table=pd.DataFrame([[t for t in X_test],y_pred, y_test]).transpose()
      table.columns = ['Tweet', 'Predicted Category', 'True Category']
      print("Percent Correct: %.2f" % (sum(table['Predicted Category'] == table['True Category'])/len(table['True Category'])))
      return table
    
    y_pred = [classify_rb(tweet) for tweet in X_test] # a list of predictions
    show_pred(y_test,y_pred)
    
        # The following is to calculate the precision, recall and F1 for a single category. 

    def evaluate(y_test,y_pred, c):
    
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0
        print (len(y_test),len(y_pred))

    for index,(true_category, predicted_category) in enumerate(zip(y_test,y_pred)):
      if true_category == c:
        if (true_category == predicted_category):
               true_positives += 1
           else:
               false_negatives += 1
           else:
           if predicted_category == c:
               false_positives += 1
           else:
               true_negatives +=1
 
     if true_positives == 0:
         precision = 0.0
         recall = 0.0
         f1 = 0.0
     else:
         precision = true_positives / (true_positives + false_positives)
         recall = true_positives / (true_positives + false_negatives)
         f1 = 2 * precision * recall / (precision + recall)
 
     print(c)
     print("Precision:", precision)
     print("Recall:", recall)
     print("F1:", f1)
     print()
 
     return f1
 
     # The following are predictions for the entire dataset. 
     y_pred = [classify_rb(tweet) for tweet in X_test]
     food_f1 = evaluate(y_test,y_pred, "Food")
     water_f1 = evaluate(y_test,y_pred, "Water")
     energy_f1 = evaluate(y_test,y_pred, "Energy")
     medical_f1 = evaluate(y_test,y_pred, "Medical")
     none_f1 = evaluate(y_test,y_pred, "None")
 
     # Compute F1 Score
     average_f1 = (food_f1 + energy_f1 + medical_f1 + water_f1 + none_f1) / 5
     print("Average F1:", average_f1)
 
     # Test the computational method
    print(classification_report(y_test, y_pred, target_names=['Energy', 'Food', 'Medical', 'None', 'Water']))

    # DataSet Implementation 

    %load_ext autoreload
    %autoreload 2
    from collections import Counter
    from random import sample
    from importlib.machinery import SourceFileLoader
    import numpy as np
    from os.path import join
    import warnings
    warnings.filterwarnings("ignore")

    import nltk
    nltk.download('punkt')
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.porter import *
    from nltk.corpus import stopwords
    nltk.download('stopwords' ,quiet=True)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn import metrics

    import nltk
    import string
    nltk.download('stopwords')
    
    nltk.download('wordnet')

    # Data load
    disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")
    
    # Filter for Profanity 
    !pip install better_profanity
    
    from better_profanity import profanity
    
    disaster_tweets['contains_profanity'] = disaster_tweets['text'].apply(lambda x: profanity.contains_profanity(x))
    disaster_clean = disaster_tweets[disaster_tweets['contains_profanity'] == False]
    disaster_tweets = disaster_clean

    disaster_tweets.head(10)

    tweets = disaster_tweets['text'].str.lower()
    tweets = tweets.apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ',x))

    tweet_labels = disaster_tweets['category']

    # Now lets tokenize the sentences Lets use an example here: 

    tweet = "Why are the wind speeds around me so high?" 
    for i in word_tokenize(tweet):
    print(i)

    # We can also STEM Words

    stemmer = PorterStemmer()
    word = "categories" #@param {type:"string"}
    print(stemmer.stem(word))

    # We also need to lemantize these words
    nltk.download('omw-1.4')
    lemma = WordNetLemmatizer()
    word = "cacti" #@param {type:"string"}
    print(lemma.lemmatize(word))

    # We also need to look out for StopWords
    eng_stopwords = stopwords.words('english')
    for word in sample(eng_stopwords, 10):
    print(word)

    word = ""
    if not word:
    print('Please enter a word')
    elif word.lower().strip() in stopwords.words('english'):
    print('Yes')
    else:
    print('No')

    stopword_set = set(stopwords.words('english'))

    def remove_stopwords(token_list):
    filtered_sentences = []
      for tweet in None:
        new_tweet = []
        for word in None:
          if word not in None:
            new_tweet.append(None)
        filtered_sentences.append(None)
      return filtered_sentences

    # Next we need to convert the code we have into numerical form: 

    d = {'I': 1, 'am': 2, 'hungry': 3, 'need': 4, 'food': 5}
    print('{:<12}|{:>2}'.format('word', 'value'))
    print('-------------------')
    for k,v in d.items(): print('{:<12}|{:>3}'.format(k,v))

    # An example and trial of a conersion would include based on the previous lines of code: 
    print('{:^10}|{:^7}|{:^8}|{:^2}|{:^9}'.format('I', 'am', 'hungry','need','food'))
    print('---------------------------------------------------')
    print('{:^10}|{:^7}|{:^8}|{:^4}|{:^9}'.format('1', '0', '0','0','0'))
    print('{:^10}|{:^7}|{:^8}|{:^4}|{:^9}'.format('0', '1', '0','0','0'))
    print('{:^10}|{:^7}|{:^8}|{:^4}|{:^9}'.format('0', '0', '1','0','0'))
    print('{:^10}|{:^7}|{:^8}|{:^4}|{:^9}'.format('0', '0', '0','1','0'))
    print('{:^10}|{:^7}|{:^8}|{:^4}|{:^9}'.format('0', '0', '0','0','1'))

# The following is also a method for helping with the encoding process
    
    def one_hot_encoding(sentences, sentence, print_word_dict = False):

    words_list = []
    words_list.extend(word_tokenize(sent))
    words_list = list(set(words_list)) # Hint: See Sets above!
    words_map_dict = {}

    for idx, w in enumerate(words_list):
    words_map_dict[w] = idx
    sent_encoding = np.zeros(len(words_list)) 
  
    for word in word_tokenize(sentence):
      if word in words_map_dict:
        print(word)
        idx = words_map_dict[word]
        sent_encoding[idx] = 1

    if print_word_dict:
      print ("Word Dictionary: ", words_map_dict)

    return sent_encoding

    # We need to know build a count function to give statistics about every tweet. 

    # The words will now be counted 
    word_count = Counter()
    for tweet in tweets[:3]:
      for t in word_tokenize(tweet):
        word_count[t]+=1
    word_count_list = [(k,v) for k,v in word_count.items()]
    word_count_list.sort(key=lambda x:x[0])
    print('{:<12}|{:>2}'.format('word', 'position'))
    print('-------------------')
    for k,v in enumerate(word_count_list): print('{:<12}|{:>3}'.format(v[0],k))
    
    # The tokens will now be counted 
    print('{:<12}|{:>2}'.format('word', 'word_count'))
    print('-------------------')
    for k,v in word_count_list: print('{:<12}|{:>3}'.format(k,v))

    # To best fit the limits of this model and to increase efficiency, the .fit() and .transform() model must be properly utilized. 
    tweet_01 = 'The first sample tweet'
    tweet_02 = 'A second sample tweet'
    train_text = [tweet_01, tweet_02]
    print(train_text)
    
    tweet_03 = 'A third sample tweet'
        
    vectorizer = CountVectorizer()
    vectorizer.fit(train_text)
    print(vectorizer.vocabulary_)
    print(vectorizer.transform([tweet_03]))

    # We now want to implement Linear Regression into our AI Model
    tweet1 = 'We Need Food'
    label1 = "Food"
    tweet2 = 'Please Send Water'
    label2 = 'Water'
    tweet3 = 'We Are Very Thirsty, We Need Water'
    label3 = 'Water'
    
    train_tweets = [tweet1, tweet2]
    train_tweets_label = [label1, label2]
    test_tweets = [tweet3]
    test_tweets_label = [label3]

    # Transform into Vector Form 
    vectorizer = CountVectorizer()
    train_vect = vectorizer.fit_transform(train_tweets) 
    model = LogisticRegression()
    model.fit(train_vect, train_tweets_label) 

    test_vect = vectorizer.transform(test_tweets)
    result = model.predict(test_vect)
    print('Actual Category: {}\nPredicted Category: {}'.format(label3, result[0]))

    # Now combine this to make logistic regression the trainer for the model 
    def train_model(tweets_to_train,train_labels):

    train_tweets = [" ".join(t) for t in tweets_to_train]
    train_tweets_label = [l for l in train_labels]
  
    vectorizer = CountVectorizer() 
    train_vect = vectorizer.fit_transform(train_tweets)
  
    model = LogisticRegression() 
    model.fit(train_vect, train_tweets_label) 
  
    return model, vectorizer

    # We can also further this to the predict model overall
    def predict(tweets_to_test, vectorizer, model):

    test_tweets = [" ".join(t) for t in  tweets_to_test]
    print(test_tweets)
  
    test_vect = vectorizer.transform(test_tweets) # Use .transform to vectorize our tweets
    result = model.predict(test_vect) # Have your model predict on the vectorized tweets
  
    return result

    # Now we can combine these two functions
    model, train_countvect = train_model(X_train, y_train)

    y_pred = predict (X_test, train_countvect, model)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Result Interpretation 
    table=pd.DataFrame([[" ".join(t) for t in X_test],y_pred, y_test]).transpose()
    table.columns = ['Tweet', 'Predicted Category', 'True Category']
    print("Percent Correct: %.2f" % (sum(table['Predicted Category'] == table['True Category'])/len(table['True Category'])))
    table

    def plot_confusion_matrix(y_true,y_predicted):
      cm = metrics.confusion_matrix(y_true, y_predicted)
      print ("Plotting the Confusion Matrix")
      labels = ['Energy', 'Food', 'Medical', 'None', 'Water']
      df_cm = pd.DataFrame(cm,index =labels,columns = labels)
      fig = plt.figure()
      res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
      plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
      plt.title('Confusion Matrix - TestData')
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.show()
      plt.close()

    plot_confusion_matrix(y_test,y_pred)

    # Prints out the accuracy 
    print('The total number of correct predictions are: {}'.format(sum(table['Predicted Category'] == table['True Category'])))
    print('The total number of incorrect predictions are: {}'.format(sum(table['Predicted Category'] != table['True Category'])))

    print('Accuracy on the test data is: {:.2f}%'.format(metrics.accuracy_score(y_test, y_pred)*100))

    # We can further expand this to learning about the characteristic of the tweet in the form of how it's words fit together in it's overall message. 
    import re
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from torchtext.vocab import GloVe
    from sklearn.model_selection import train_test_split

    !wget http://nlp.uoregon.edu/download/embeddings/glove.6B.300d.txt # This is going to be the source of the Glove Vectors from Stanford University
    
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import matplotlib
    import matplotlib.pyplot as plt
    import requests, io, zipfile

    # First we need to load and filter the data
    disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")
    
    #profanity filter:
    !pip install better_profanity
    
    from better_profanity import profanity
    
    disaster_tweets['contains_profanity'] = disaster_tweets['text'].apply(lambda x: profanity.contains_profanity(x))
    disaster_clean = disaster_tweets[disaster_tweets['contains_profanity'] == False]
    disaster_tweets = disaster_clean

    disaster_tweets.head()

    # We now need to extract tweets and their labels 
    tweets = disaster_tweets['text'].str.lower()
    tweets = tweets.apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ',x))

    tweet_labels = disaster_tweets['category']

    # Split it into train and test set
    X_train, X_test, y_train, y_test = train_test_split(tweets, tweet_labels, test_size=0.2, random_state=1,stratify = tweet_labels)

    VEC_SIZE = 300
    glove = GloVe(name='6B', dim=VEC_SIZE)

    def get_word_vector(word):
        try:
          return glove.vectors[glove.stoi[word.lower()]].numpy()
        except KeyError:
          return None

    # We can now compare the similarities of words

    def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    word1 = "mumbai" 
    word2 = "delhi" 
    
    print('Word 1:', word1)
    print('Word 2:', word2)
    
    def cosine_similarity_of_words(word1, word2):
      vec1 = get_word_vector(word1)
      vec2 = get_word_vector(word2)
    
      if vec1 is None:
        print(word1, 'is not a valid word. Try another.')
      if vec2 is None:
        print(word2, 'is not a valid word. Try another.')
      if vec1 is None or vec2 is None:
        return None
    
      return cosine_similarity(vec1, vec2)

    print('\nCosine similarity:', cosine_similarity_of_words(word1, word2))

    # We are also able to Average out the Glove Function 

    def glove_transform_data_descriptions(data):
    X = np.zeros((len(data), VEC_SIZE))


    for i, tweet in enumerate(data):
        found_words = 0.0
        tweet = tweet.strip()
        for word in tweet.split():

            vec = get_word_vector(word)
            if vec is not None:
                # Increment found_words and add vec to X[i].
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words

    return X

    glove_train_X = glove_transform_data_descriptions(X_train)
    glove_train_y = [label for label in y_train]
    print ("Train_shape",glove_train_X.shape)
    glove_test_X = glove_transform_data_descriptions(X_test)
    glove_test_y = [label for label in y_test]
    print ("Test shape",glove_test_X.shape)

    # We can also include the addition of Linear Regression into our model 
    model = LogisticRegression()
    model.fit(glove_train_X, glove_train_y)
    
    glove_train_y_pred = model.predict(glove_train_X)
    print('Train accuracy', accuracy_score(glove_train_y, glove_train_y_pred))
    
    glove_test_y_pred = model.predict(glove_test_X)
    print('Val accuracy', accuracy_score(glove_test_y, glove_test_y_pred))
    
    print('Confusion matrix:')
    print(confusion_matrix(glove_test_y, glove_test_y_pred))
    
    prf = precision_recall_fscore_support(glove_test_y, glove_test_y_pred)
    
    print('Precision:', prf[0][1])
    print('Recall:', prf[1][1])
    print('F-Score:', prf[2][1])
        
    # Now it's time for evaluation 

    # Lets first find the wrong tweets
    pd.set_option('max_colwidth', 500)
    incorrect_tweets = []
    incorrect_y_test = []
    incorrect_y_pred = []
    for (t,x,y) in zip(X_test,y_test,glove_test_y_pred):
    if x != y:
      incorrect_tweets.append(t)
      incorrect_y_test.append(x)
      incorrect_y_pred.append(y)

    table=pd.DataFrame([incorrect_tweets,incorrect_y_pred,incorrect_y_test]).transpose()
    table.columns = ['Tweet', 'Predicted Category', 'True Category']

    # Check out the visualized embeddings model involving t-SNE
    from gensim.models.word2vec import Word2Vec
    from sklearn.manifold import TSNE
    import plotly.express as px
    
    import re
    import matplotlib.pyplot as plt
    
    def clean(text):
        """Remove posting header, split by sentences and words, keep only letters"""
        lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '', re.sub('\n', ' ', text)))
        return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]
    
    sentences = [line for text in tweets for line in clean(text)]
    
    model = Word2Vec(sentences, workers=4, vector_size=100, min_count=30, window=10, sample=1e-3)
    
    def tsne_plot(model):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []
    
        for word in model.wv.key_to_index:
            tokens.append(model.wv.get_vector(word, norm=True))
            labels.append(word)
    
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(np.array(tokens))
    
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
    
        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()
    
    def tsne_plot_3d(model):
        "Creates a TSNE model and plots it in 3D"
        labels = []
        tokens = []
    
        for word in model.wv.key_to_index:
            tokens.append(model.wv.get_vector(word, norm=True))
            labels.append(word)
    
        tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(np.array(tokens))
    
        x = []
        y = []
        z = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
    
        fig = px.scatter_3d(x=x, y=y, z=z, text=labels)
        fig.update_traces(textposition='top center')
        fig.update_layout(title_text='GloVe Word Embeddings in 3D')
        fig.show()

    # We now can display this 3D model of the word vectors we made with t-SNE
      tsne_plot(model)
