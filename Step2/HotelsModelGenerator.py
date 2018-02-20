import argparse
import pandas as pd
import sys
from Step2.Classifiers.DecisionTreeHotelsClassifier import DecisionTreeHotelsClassifier
from Step2.Classifiers.SVMHotelsClassifier import SVMHotelsClassifier
from Step2.Classifiers.NaiveBayesHotelsClassifier import GaussianNBHotelsClassifier


def get_classifier(args):
    if args.classifier == 'DecisionTree':
        return DecisionTreeHotelsClassifier(min_samples_split=args.min_samples_split)
    elif args.classifier == 'NaiveBayes':
        return GaussianNBHotelsClassifier()
    return SVMHotelsClassifier()


def get_training_data(csv_path, percentage_size):
    hotels_df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    if not hotels_df.empty:
        training_df = hotels_df.head(int(len(hotels_df)*percentage_size))
        training_x = training_df[['Hotel Name', 'Weekday', 'Snapshot Date', 'Checkin Date', 'DayDiff']]
        training_y = training_df['Discount Code']
        return training_x, training_y
    return None


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hotels Model Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to manipulate", type=str)
    parser.add_argument('--output_model_path', required=True, help="Where to export the created model", type=str)
    parser.add_argument('--train_size', required=False,
                        help="Percentage training size from the csv",
                        type=restricted_float,
                        default=0.8)
    subparsers = parser.add_subparsers(dest='classifier', description='Classifier type to work with')
    dt_parser = subparsers.add_parser('DecisionTree')
    nb_parser = subparsers.add_parser('NaiveBayes')
    svm_parser = subparsers.add_parser('SVM')
    dt_parser.add_argument('--min_samples_split',
                           help='Minimum samples split for a new branch node',
                           type=int,
                           default=20)

    args = parser.parse_args(sys.argv[1:])

    # try:
    print('Getting classifier for ' + args.classifier)
    classifier = get_classifier(args)
    print('Getting training set')
    training_x, training_y = get_training_data(args.input_csv, args.train_size)
    print('Training classifier')
    classifier.train(training_x, training_y)
    print('Saving classifier to ' + args.output_model_path)
    classifier.save_model(args.output_model_path)
    # except Exception as e:
    #     print(str(e))
