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
from sklearn.base import BaseEstimator, TransformerMixin


# Borrowed from GridSearch Pipeline exercise
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name="Messages", con=engine)
    X = df.message
    Y = df.as_matrix(columns=df.columns[4:])
    categories = list(df.columns[4:])
    return X, Y, categories


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_words


def build_model(parameters=None):

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
    outfile = open(model_filepath, "wb")
    pickle.dump(model, outfile)
    outfile.close()


def main():
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
