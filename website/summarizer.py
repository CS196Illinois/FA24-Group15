import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import nltk
import spacy
#import torch
#from transformers import BartForConditionalGeneration, BartTokenizer
nltk.download('punkt')

#def preprocess_text(text):
#    sentences = sent_tokenize(text)
#    return sentences

#def sentence_similarity2(sent1, sent2): # not planning to use; change back to just sentence_similarity
#    words1 = word_tokenize(sent1.lower())
#    words2 = word_tokenize(sent2.lower())

#    all_words = list(set(words1).union(set(words2)))

#    vec1 = [words1.count(word) for word in all_words]
#    vec2 = [words2.count(word) for word in all_words]

#    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

#def textrank(sentences, top_n=3): # designing our own
#    similarity_matrix = np.zeros((len(sentences), len(sentences)))

#    for i in range(len(sentences)):
#        for j in range(len(sentences)):
#            if i != j:
#                similarity_matrix[i][j] = sentence_similarity2(sentences[i], sentences[j])

#    nx_graph = nx.from_numpy_array(similarity_matrix)
#    scores = nx.pagerank(nx_graph)

#    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
#    top_n = min(top_n, len(ranked_sentences))
#    return [ranked_sentences[i][1] for i in range(top_n)]

# sampleText1 =
# Comprehensive and scholarly, this well-designed and class-tested text presents Greek and Roman myths in a lively and easy-to-read manner. It features fresh translations, numerous illustrations (ancient and modern) of classical myths and legends, and commentary that emphasizes the anthropological, historical, religious, sociological, and economic contexts in which the myths were told. This book covers myths of creation, myths of fertility, myths of the Olympians, Heracles, Oedipus, Trojan War, Roman Myth, Odysseus, and more. It also introduces students to classic literary works by Homer, Hesiod, Aeschylus, Sophocles, Euripides, and Ovid. For anyone interested in learning more about the creation and modern interpretation of classical myths.
# sampleText2 = 
# The National Football League (NFL) is a professional American football league that consists of 32 teams, divided equally between the American Football Conference (AFC) and the National Football Conference (NFC). The NFL is one of the major professional sports leagues in the United States and Canada and the highest professional level of American football in the world.[5] Each NFL season begins annually with a three-week preseason in August, followed by the 18-week regular season, which runs from early September to early January, with each team playing 17 games and having one bye week. Following the conclusion of the regular season, seven teams from each conference, including four division winners and three wild card teams, advance to the playoffs, a single-elimination tournament, which culminates in the Super Bowl, played in early February between the winners of the AFC and NFC championship games.
# sentences = preprocess_text(sampleText2)
# summary = textrank(sentences, top_n=3)
# print("Summary:")
# for sentence in summary:
#   print(sentence)


# UPDATED CODE STARTS HERE

nlp = spacy.load("en_core_web_sm")

# sentence similarity score calculation through spaCy, not bag of words
def sentence_similarity2(sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    similarity = doc1.similarity(doc2)

    return similarity

# creates a matrix of similarity scores; each sentence similarity to itself = 0
# first sentence of a summary of text is located at index 0
def createMatrix(text):
  listOfSentences = sent_tokenize(text)
  numberOfSentences = len(listOfSentences)
  matrix = np.zeros((numberOfSentences, numberOfSentences))
  for i in range(numberOfSentences):
    for j in range(numberOfSentences):
      if i != j:
        matrix[i][j] = sentence_similarity2(listOfSentences[i], listOfSentences[j])
  return matrix

# generates sub-bullet point sentences and returns them as an array
def subBulletPts(index, fullText, threshold = 0.75):
  similarityMatrix = createMatrix(fullText) # to find the most similar sentence
  listOfSentences = sent_tokenize(fullText) # for finding 2-3 sub bullet point sentences (use index)
  arrayMainSent = similarityMatrix[index] # one row, displays similarity scores of all other sentences to main sentence
  arrayToReturn = []
  #print("Sub bullet points for sentence: " + listOfSentences[index])
  for i in range(len(arrayMainSent)):
    if arrayMainSent[i] > threshold:
      #print(listOfSentences[i]) # prints sentences that (compared to main bullet pt) have a similarity score > 0.75
      arrayToReturn.append(listOfSentences[i])
  return arrayToReturn

# optional index finder method
def indexFinder(fullText, mainSent):
  listOfSentences = sent_tokenize(fullText)
  for i in range(len(listOfSentences)):
    if listOfSentences[i] == mainSent:
      return i

# Paraphrase function using torch: COMMENT OUT IF CAN'T DOWNLOAD PACKAGES
#def paraphrase(input_sentence):
#  model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
#  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  model = model.to(device)
#  tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
#  batch = tokenizer(input_sentence, return_tensors='pt')
#  generated_ids = model.generate(batch['input_ids'])
#  generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#  return generated_sentence

# Ethan's code is here
#adjMatrix = createMatrix(text)

def bestFit(text, varThreshold):

    '''We need to experiment with varThreshold values, or add an iterative
    method in some way.'''
    listOfSentences = sent_tokenize(text)
    adjMatrix = createMatrix(text)
    numberOfSentences = len(listOfSentences)
    avgTransform = [1 / (numberOfSentences - 1)] * numberOfSentences
    avgMeasures = np.dot(adjMatrix, avgTransform)
    toReturn = []
    for i in range(numberOfSentences):
      sumMeasure = 0
    for j in range(numberOfSentences):
        if (adjMatrix[i][j] - avgMeasures[i]) ** 2 > varThreshold:
            adjMatrix[i][j] = 0
            sumMeasure += adjMatrix[i][j]
        toReturn.append(sumMeasure)
    return toReturn # retrieve largest 3 scores using max

# retrieve three largest scores and save their indices in a list
def mainBulletPts(text):
  avgScoreList = bestFit(text, 1) # assuming varThreshold = 1 (default)
  sentToAvgScore = {} # key = sentence index, value = score
  for i in range(len(avgScoreList)):
    sentToAvgScore[i] = avgScoreList[i]
  listOfValues = list(sentToAvgScore.values())
  listOfKeys = list(sentToAvgScore.keys())
  listOfIndices = [] # holds 3 main bullet point sentences' index
  for i in range(3):
     listOfIndices.add(listOfKeys[listOfValues.index(max(listOfValues))])
  return listOfIndices




#Right now, this returns a list of values whose indices correspond to sentences.
#The sentences with the largest values are what we want.

# code Daniel uses to generate sub-bullet points
def generate_elaboration(bullet_point):
    elaboration = ""
    
    # Example elaboration based on keywords (you can expand this with AI models like GPT)
    if "technology" in bullet_point.lower():
        elaboration = "This advancement is pushing boundaries in fields like AI, automation, and sustainability."
    elif "environment" in bullet_point.lower():
        elaboration = "Environmental awareness is critical in addressing issues like climate change and resource conservation."
    else:
        elaboration = "This topic is crucial in shaping future developments in its respective field."
    
    return elaboration

def summarize(input):
    sentences = preprocess_text(input)
    summary = textrank(sentences, top_n=3)  
     # Creating the final bullet points with elaborations
    expanded_bullet_points = []
    for point in summary:
      elaboration = generate_elaboration(point)  # Add elaboration to the bullet point
        # Create a nested bullet point structure with elaboration
        #expanded_bullet_points.append(f"• {point} \n    - {elaboration}")
      expanded_bullet_points.append(f"<ul><li>{point}<ul><li>{elaboration}</li></ul></li></ul>")
    
    return expanded_bullet_points


