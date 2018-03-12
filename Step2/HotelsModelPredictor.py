import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, auc
import sys
from Step2.Classifiers.ModelClassifierFactory import ModelClassifierFactory
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

FEATURES_LIST = ['Hotel Name', 'Weekday', 'Snapshot Date', 'Checkin Date', 'DayDiff']
TARGET_CLASS = 'Discount Code'
COMBINED_LIST = ['Hotel Name', 'Weekday', 'Snapshot Date', 'Checkin Date', 'DayDiff', 'Discount Code']
DISCOUNT_CODE_CLASSES = [1, 2, 3, 4]


def split_training_set(training_set):
    return training_set[FEATURES_LIST], \
           training_set[TARGET_CLASS]


def print_prediction_stats(target_y, pred_y):
    print(len([x for x in target_y - pred_y if x == 0]) / len(pred_y))
    MSE = mean_squared_error(target_y, pred_y)
    CM = confusion_matrix(target_y, pred_y)
    print('######################################')
    print('######################################')
    print('Confusion Matrix')
    print(CM)
    print('######################################')
    print('######################################')
    print('MSE')
    print(MSE)
    print('######################################')
    print('######################################')


def plot_roc_curve_onevsrest(target_y, pred_y):
    target_y = label_binarize(target_y, DISCOUNT_CODE_CLASSES)
    pred_y = label_binarize(pred_y, DISCOUNT_CODE_CLASSES)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(DISCOUNT_CODE_CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(target_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target_y.ravel(), pred_y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(DISCOUNT_CODE_CLASSES))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(DISCOUNT_CODE_CLASSES)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 4

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(DISCOUNT_CODE_CLASSES)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def training_flow(args):
    # Get the fitting classifier for the args
    factory_singleton = ModelClassifierFactory()
    print('Getting classifier for ' + args.classifier)
    classifier = factory_singleton.create_classifier(args.classifier)(**vars(args))

    # Note that if the one vs rest mode is enabled
    # The classifier is wrapped with one vs rest classifier
    if args.enable_one_vs_rest:
        print('Using one vs rest wrapper')
        classifier = factory_singleton.create_classifier('OneVsRest')(classifier)

    # Load the CSV and take the needed columns
    hotels_df = pd.read_csv(args.input_csv, encoding="ISO-8859-1")
    hotels_df = hotels_df[COMBINED_LIST]

    # Fit the labelizer and transform the needed columns with split for training and validation
    classifier.fit_labelizer(hotels_df)
    print('Getting training set')
    training_set, validation_set = split_training_test(hotels_df, args.train_size)
    print('Encoding Data')
    labeled_training_set = classifier.transform_labels(training_set)
    labeled_validation_set = classifier.transform_labels(validation_set)

    # Split the data to train and validation
    print('Splitting data')
    labeled_training_x, labeled_training_y = split_training_set(training_set=labeled_training_set)
    labeled_validation_x, labeled_validation_y = split_training_set(training_set=labeled_validation_set)

    # Train and save the model
    classifier.train(labeled_training_x, labeled_training_y)
    print('Saving classifier to ' + args.output_model_folder)
    factory_singleton.save_classifier(classifier, args.output_model_folder)

    # This is for validation, and stats computation of the created model
    predictions = classifier.predict(labeled_validation_x)
    print_prediction_stats(labeled_validation_y, predictions)
    plot_roc_curve_onevsrest(labeled_validation_y, predictions)


def predict_flow(args):
    # Load the saved model and labelizers from the file system
    classifier_folder = args.input_model_folder
    factory_singleton = ModelClassifierFactory()
    print('Getting classifier from folder ' + classifier_folder)
    classifier = factory_singleton.load_classifier(classifier_folder)

    # Read and label the data
    print('Reading CSV')
    hotels_df_to_predict = pd.read_csv(args.input_csv, encoding="ISO-8859-1")
    hotels_df_to_predict = hotels_df_to_predict[FEATURES_LIST]
    print('Labeling CSV')
    # Note that the copy is required for saving the csv later on due to the labelizer
    labeled_hotels_df_to_predict = classifier.transform_labels(hotels_df_to_predict.copy())

    # Predict
    print('Trying to predict...')
    predictions = classifier.predict(labeled_hotels_df_to_predict)
    print('Results:')
    print(predictions)

    # Save the results to the csv
    if args.save_predictions_to_csv:
        hotels_df_to_predict[TARGET_CLASS] = predictions
        hotels_df_to_predict.to_csv(args.input_csv)


def split_training_test(hotels_df, percentage_size):
    training_df = hotels_df.head(int(len(hotels_df) * percentage_size))
    test_df = hotels_df.tail(int(len(hotels_df) * (1 - percentage_size)))
    return training_df, test_df


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hotels Model Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to manipulate", type=str)
    subparsers = parser.add_subparsers(dest='mode', description='Mode of the script')
    training_parser = subparsers.add_parser('train')
    predict_parser = subparsers.add_parser('predict')

    training_parser.add_argument('--output_model_folder', required=True, help="Where to export the created model",
                                 type=str)
    training_parser.add_argument('--train_size', required=False,
                                 help="Percentage training size from the csv",
                                 type=restricted_float,
                                 default=0.5)
    training_parser.add_argument('--enable_one_vs_rest', required=False,
                                 help='If specified, one vs rest classifier wrapper will be used',
                                 action='store_true')
    training_subparsers = training_parser.add_subparsers(dest='classifier', description='Classifier type to work with')
    dt_parser = training_subparsers.add_parser('DecisionTree')
    nb_parser = training_subparsers.add_parser('NaiveBayes')
    svm_parser = training_subparsers.add_parser('SVM')
    dt_parser.add_argument('--min_samples_split',
                           help='Minimum samples split for a new branch node',
                           type=int,
                           default=20)

    predict_parser.add_argument('--input_model_folder',
                                type=str,
                                help='The trained model to try to predict with')
    predict_parser.add_argument('--save_predictions_to_csv',
                                help='Whether to save the predictions to the input CSV or not',
                                required=False,
                                action='store_true')

    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'train':
        training_flow(args)
    elif args.mode == 'predict':
        predict_flow(args)
