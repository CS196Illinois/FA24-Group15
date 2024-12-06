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

# sampleText1 = "Comprehensive and scholarly, this well-designed and class-tested text presents Greek and Roman myths in a lively and easy-to-read manner. It features fresh translations, numerous illustrations (ancient and modern) of classical myths and legends, and commentary that emphasizes the anthropological, historical, religious, sociological, and economic contexts in which the myths were told. This book covers myths of creation, myths of fertility, myths of the Olympians, Heracles, Oedipus, Trojan War, Roman Myth, Odysseus, and more. It also introduces students to classic literary works by Homer, Hesiod, Aeschylus, Sophocles, Euripides, and Ovid. For anyone interested in learning more about the creation and modern interpretation of classical myths."
# sampleText2 = "The National Football League (NFL) is a professional American football league that consists of 32 teams, divided equally between the American Football Conference (AFC) and the National Football Conference (NFC). The NFL is one of the major professional sports leagues in the United States and Canada and the highest professional level of American football in the world.[5] Each NFL season begins annually with a three-week preseason in August, followed by the 18-week regular season, which runs from early September to early January, with each team playing 17 games and having one bye week. Following the conclusion of the regular season, seven teams from each conference, including four division winners and three wild card teams, advance to the playoffs, a single-elimination tournament, which culminates in the Super Bowl, played in early February between the winners of the AFC and NFC championship games."

nlp = spacy.load("en_core_web_sm")
def sentence_similarity2(sentence1, sentence2): # sentence similarity through spaCy, not bag of words
    # Process the sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    # Calculate similarity
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


# optional index finder method
def indexFinder(fullText, mainSent):
  listOfSentences = sent_tokenize(fullText)
  for i in range(len(listOfSentences)):
    if listOfSentences[i] == mainSent:
      return i
    
def bestFit(text):
    '''We need to experiment with varThreshold values, or add an iterative
    method in some way.'''
    listOfSentences = sent_tokenize(text)
    adjMatrix = createMatrix(text)
    numberOfSentences = len(listOfSentences)
    avgTransform = [1 / (numberOfSentences - 1)] * numberOfSentences
    avgMeasures = np.dot(adjMatrix, avgTransform)
    scoreList = []
    for i in range(numberOfSentences):
      sumMeasure = 0
      for j in range(numberOfSentences):
          if (adjMatrix[i][j] - avgMeasures[i]) ** 2 > 0.3:
              adjMatrix[i][j] = 0
          sumMeasure += adjMatrix[i][j]
      scoreList.append(sumMeasure)
    toReturn = [] # sentences
    for i in range(3):
       maxValue = max(scoreList)
       index = scoreList.index(maxValue)
       toReturn.append(listOfSentences[index])
       scoreList[index] = 0
    return toReturn # this is list of 3 main bullet points

#code Daniel uses to generate sub-bullet points
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
#new one
# generates sub-bullet point sentences and returns them as an array
def subBulletPts(mbpIndex, text, threshold = 0.75):
  bulletPoints = bestFit(text)
  similarityMatrix = createMatrix(text) # to find the most similar sentence
  listOfSentences = sent_tokenize(text) # for finding 2-3 sub bullet point sentences (use index)
  index = listOfSentences.index(bulletPoints[mbpIndex])
  #for i in range(3):
  #   listOfSentences.remove(bulletPoints[i])
  arrayMainSent = similarityMatrix[index] # one row, displays similarity scores of all other sentences to main sentence
  arrayToReturn = []
  for i in range(len(arrayMainSent)):
    if arrayMainSent[i] > threshold and listOfSentences[i] not in bulletPoints:
      #print(listOfSentences[i]) # prints sentences that (compared to main bullet pt) have a similarity score > 0.75
      arrayToReturn.append(listOfSentences[i])
  return arrayToReturn


def summarize(input):
    summary =  bestFit(input)
    expanded_bullet_points = []
    for point in summary:
      elaboration = generate_elaboration(input)
      # expanded_bullet_points.append(f"<ul><li>{point}<ul><li>{elaboration}</li></ul></li></ul>")
      expanded_bullet_points.append(f"<ul><li>{point}</li></ul>")
    return expanded_bullet_points

# # updated1 one
# def summarize(input):
#     #summary = textrank(sentences, top_n=3)  
#     # Generate bullet points using mainBulletPts
#     # Extract the corresponding sentences from the preprocessed list
#     summary =  bestFit(input)
#     # Creating the final bullet points with elaborations
#     expanded_bullet_points = []
#     for i in range(3):
#       elaborationlist = subBulletPts(i, input)
#       for elaboration in elaborationlist:
#          expanded_bullet_points.append(f"<ul><li>{summary[i]}<ul><li>{elaboration}</li></ul></li></ul>")
#     return expanded_bullet_points

