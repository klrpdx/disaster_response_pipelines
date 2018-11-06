import sys
import re
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """
    Load the data from the database
    :param database_filepath: The location of the database file
    :return: The messages, the labels, and the categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name="Messages", con=engine)
    X = df.message
    Y = df.as_matrix(columns=df.columns[4:])
    categories = list(df.columns[4:])
    return X, Y, categories


def tokenize(text):
    """
    Process the text. Convert to lowercase, tokenize, and remove stop words
    :param text: The text to process
    :return: The tokenized text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_words


def build_model(parameters=None):
    """
    Build the model. Optionally run GridSearch on the provided parameters
    :param parameters: Parameters to search on with GridSearch. Optional.
    :return: The model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=100, min_samples_leaf=1)))
    ])

    if parameters:
        model = GridSearchCV(pipeline, param_grid=parameters)
        return model
    else:
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model. Predict for the test messages and iterate over each category name and run
    classification_report on the results.
    :param model: The model to evaluate
    :param X_test: The set of test messages
    :param Y_test: The set of test labels
    :param category_names: The category names
    :return: Dictionary of the results. Can be used for comparing the performance of different models
    """
    results = {}
    predicted = model.predict(X_test)

    for i, column in enumerate(category_names):
        pred_value = [item[i] for item in predicted]
        true_value = [item[i] for item in Y_test]
        print("Category: {}".format(column))
        print(classification_report(true_value, pred_value))
        results[column] = classification_report(true_value, pred_value, output_dict=True)

    return results


def evaluate_results(results_set1, results_set2, category_names):
    """
    Compare two models. Prints out the difference in precision, recall and f1 scores between two models. The
    results are reported as score for model 2 - score for model 1. Therefore, a positive number means model 2
    has improved results.
    :param results_set1: The dictionary of results produced by evaluate_model on the first model
    :param results_set2: The dictionary of results produced by evaluate_model on the second model
    :param category_names: The category names to iterate over
    :return:
    """
    model1_scores = {}
    model2_scores = {}
    empty = {'recall': 0, 'precision': 0, 'f1-score': 0}

    for category in category_names:
        model1_scores['0'] = results_set1[category].get('0', empty)
        model1_scores['1'] = results_set1[category].get('1', empty)

        model2_scores['0'] = results_set2[category].get('0', empty)
        model2_scores['1'] = results_set2[category].get('1', empty)

        if results_set1[category].get('0') is None:
            print("")
            print("No results for set1-0, category {}".format(category))
            print("")

        if results_set1[category].get('1') is None:
            print("")
            print("No results for set1-1, category {}".format(category))
            print("")

        if results_set2[category].get('0') is None:
            print("")
            print("No results for set2-0, category {}".format(category))
            print("")

        if results_set2[category].get('1') is None:
            print("")
            print("No results for set2-1, category {}".format(category))
            print("")

        print('{}: change from model 1 to model 2:'.format(category))
        print('\t Recall-0: {:1.2f}'.format(model2_scores['0']['recall'] - model1_scores['0']['recall']))
        print('\t Recall-1: {:1.2f}'.format(model2_scores['1']['recall'] - model1_scores['1']['recall']))

        print('\t Precision-0: {:1.2f}'.format(model2_scores['0']['precision'] - model1_scores['0']['precision']))
        print('\t Precision-1: {:.2f}'.format(model2_scores['1']['precision'] - model1_scores['1']['precision']))

        print('\t f1-score-0: {:1.2f}'.format(model2_scores['0']['f1-score'] - model1_scores['0']['f1-score']))
        print('\t f1-score-1: {:1.2f}'.format(model2_scores['1']['f1-score'] - model1_scores['1']['f1-score']))


def save_model(model, model_filepath):
    """
    Save the model
    :param model: The model to save
    :param model_filepath: The filepath to save the model at
    """
    outfile = open(model_filepath, "wb")
    pickle.dump(model, outfile)
    outfile.close()


def main():
    """
    Main method to run all the steps for loading the data and building, training, evaluating and saving the model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
