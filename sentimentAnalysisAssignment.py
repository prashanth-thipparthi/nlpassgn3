import math
import nltk

def readFile(fileName):
    fileOpen = open(fileName,'r',encoding='utf-8')
    fileData = fileOpen.readlines()
    return fileData

def prepare_training_and_test_data(pos_example_list, neg_example_list):
    pos_training_data = pos_example_list[:int(len(pos_example_list)*0.8)]
    neg_training_data = neg_example_list[:int(len(neg_example_list)*0.8)]

    pos_test_data = pos_example_list[int(len(pos_example_list)*0.8):]
    neg_test_data = neg_example_list[int(len(neg_example_list)*0.8):]

    return pos_training_data, neg_training_data, pos_test_data, neg_test_data

def separateIDAndReview(fileData):
    idsList = []
    reviewsList = []
    for line in fileData:
        lineList = line.split('\t')
        idsList.append(lineList[0])
        reviewsList.append(lineList[1].rstrip())
    reviewsSize = len(reviewsList)
    return reviewsSize,idsList,reviewsList

def extractWordsFromReview(reviews):
    reviewWords = []
    for review in reviews:
        words = review.split()
        for word in words:
            reviewWords.append(word)
    totalNoOfReviewWords = len(reviewWords)
    return totalNoOfReviewWords,reviewWords

def wordVocabularyFromPositiveAndNegativeReviews(positiveReviewWords,negativeReviewWords):
    vocab = []
    for word in positiveReviewWords:
        if word not in vocab:
            vocab.append(word)

    for word in negativeReviewWords:
        if word not in vocab:
            vocab.append(word)

    vocabSize = len(vocab)
    return vocabSize,vocab

def WordFrequencyCount(positiveReviewWords,vocab):
    wordFreq = {}
    for word in positiveReviewWords:
        if word not in wordFreq:
            wordFreq[word] = 1
        else:
            wordFreq[word]+=1
    for word in vocab:
        if word not in wordFreq:
            wordFreq[word] = 0
    return wordFreq
def smoothedWordProbability(wordFrequency,reviewSize,vocabSize):
    word_prob = {}
    for word in wordFrequency:
        word_prob[word] = (wordFrequency[word] + 1) / (reviewSize + vocabSize)
    return word_prob

def calculatePior(posReviewSize,negReviewSize):
    positivePrior = posReviewSize / (posReviewSize + negReviewSize)
    negativePrior = negReviewSize / (posReviewSize + negReviewSize)
    return positivePrior, negativePrior

def predictSentiment(test_data, pos_word_prob, neg_word_prob, pos_prior, neg_prior, pos_word_size, neg_word_size, vocab_size):
    review_size, id_list, review_list = separateIDAndReview(test_data)
    result_list = []
    for review in review_list:
        result = predict(review, pos_word_prob, neg_word_prob, pos_prior, neg_prior, pos_word_size, neg_word_size, vocab_size)
        result_list.append(result)
    return result_list


def predict(review, pos_word_prob, neg_word_prob, pos_prior, neg_prior, pos_word_size, neg_word_size, vocab_size):

    # Calculate log prob for positive class
    pos_log_prob = calculate_class_prob(review, pos_word_prob, pos_prior, pos_word_size, vocab_size)

    # Calculate log prob for negative class
    neg_log_prob = calculate_class_prob(review, neg_word_prob, neg_prior, neg_word_size, vocab_size)

    if pos_log_prob > neg_log_prob:
        return "POS"
    else:
        return "NEG"


def calculate_class_prob(review, word_prob, prior, word_size, vocab_size):
    log_prob = math.log(prior)
    for word in review.split():
        if word in word_prob:
            log_prob += math.log(word_prob[word])
        else:
            log_prob += math.log(1 / (word_size + vocab_size))
            # print("unknown")
    return log_prob

def main():
    positiveReviewList = readFile('hotelPositiveTrain.txt')
    negativeReviewList = readFile('hotelNegativeTrain.txt')
    positiveTraining_data, negativeTraining_data, positiveTestData, negativeTestData = prepare_training_and_test_data(
        positiveReviewList, negativeReviewList)
    posReviewSize,posIdsList,posReviewsList = separateIDAndReview(positiveReviewList)
    totalNoOfPositiveReviewWords, positiveReviewWords = extractWordsFromReview(posReviewsList)


    negReviewSize, negIdsList, negReviewsList = separateIDAndReview(negativeReviewList)
    totalNoOfNegativeReviewWords, negativeReviewWords = extractWordsFromReview(negReviewsList)
    vocabSize, vocab = wordVocabularyFromPositiveAndNegativeReviews(positiveReviewWords,negativeReviewWords)

    positiveWordFrequency = WordFrequencyCount(positiveReviewWords,vocab)
    negativeWordFrequency = WordFrequencyCount(negativeReviewWords,vocab)

    positiveWordProbability = smoothedWordProbability(positiveWordFrequency,posReviewSize,vocabSize)
    negativeWordProbability = smoothedWordProbability(negativeWordFrequency, negReviewSize, vocabSize)

    positivePrior,negativePrior = calculatePior(posReviewSize,negReviewSize)

    positiveResultList = predictSentiment(positiveTestData,positiveWordProbability,negativeWordProbability,positivePrior,negativePrior,posReviewSize,negReviewSize, vocabSize)
    print(positivePrior)
    print(negativePrior)

if __name__ == '__main__':
    main()