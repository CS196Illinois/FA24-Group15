import nltk
import networkx as nx
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
#nltk.download('punkt')
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def sentence_similarity(sent1, sent2):
    words1 = word_tokenize(sent1.lower())
    words2 = word_tokenize(sent2.lower())

    all_words = list(set(words1).union(set(words2)))

    vec1 = [words1.count(word) for word in all_words]
    vec2 = [words2.count(word) for word in all_words]

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def textrank(sentences, top_n=3):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n = min(top_n, len(ranked_sentences))
    return [ranked_sentences[i][1] for i in range(top_n)]

# sampleText1 = "Comprehensive and scholarly, this well-designed and class-tested text presents Greek and Roman myths in a lively and easy-to-read manner. It features fresh translations, numerous illustrations (ancient and modern) of classical myths and legends, and commentary that emphasizes the anthropological, historical, religious, sociological, and economic contexts in which the myths were told. This book covers myths of creation, myths of fertility, myths of the Olympians, Heracles, Oedipus, Trojan War, Roman Myth, Odysseus, and more. It also introduces students to classic literary works by Homer, Hesiod, Aeschylus, Sophocles, Euripides, and Ovid. For anyone interested in learning more about the creation and modern interpretation of classical myths."
# sampleText2 = "The National Football League (NFL) is a professional American football league that consists of 32 teams, divided equally between the American Football Conference (AFC) and the National Football Conference (NFC). The NFL is one of the major professional sports leagues in the United States and Canada and the highest professional level of American football in the world.[5] Each NFL season begins annually with a three-week preseason in August, followed by the 18-week regular season, which runs from early September to early January, with each team playing 17 games and having one bye week. Following the conclusion of the regular season, seven teams from each conference, including four division winners and three wild card teams, advance to the playoffs, a single-elimination tournament, which culminates in the Super Bowl, played in early February between the winners of the AFC and NFC championship games."
# sentences = preprocess_text(sampleText2)
# summary = textrank(sentences, top_n=3)
# print("Summary:")
# for sentence in summary:
#   print(sentence)

def sentence_similarity2(sentence1, sentence2): # sentence similarity through spaCy, not bag of words
    # Process the sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    # Calculate similarity
    similarity = doc1.similarity(doc2)

    return similarity

def createMatrix(text):
  listOfSentences = sent_tokenize(text)
  numberOfSentences = len(listOfSentences)
  matrix = np.zeros((numberOfSentences, numberOfSentences))
  for i in range(numberOfSentences):
    for j in range(numberOfSentences):
      if i != j:
        matrix[i][j] = sentence_similarity2(listOfSentences[i], listOfSentences[j])
  return matrix

# function defined if provided an index, otherwise call indexFinder
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

def indexFinder(fullText, mainSent):
  listOfSentences = sent_tokenize(fullText)
  for i in range(len(listOfSentences)):
    if listOfSentences[i] == mainSent:
      return i


def summarize(input):
    sentences = preprocess_text(input)
    summary = textrank(sentences, top_n=3)  
     # Creating the final bullet points with elaborations
    expanded_bullet_points = []
    #index1 = indexFinder(input, )
    for point in summary:
        elaboration = subBulletPts(0, input, .75) # Add elaboration to the bullet point
        # Create a nested bullet point structure with elaboration
        #expanded_bullet_points.append(f"• {point} \n    - {elaboration}")
        expanded_bullet_points.append(f"<ul><li>{point}<ul><li>{elaboration}</li></ul></li></ul>")
    
    return expanded_bullet_points


