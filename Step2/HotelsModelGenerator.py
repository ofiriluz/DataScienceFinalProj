import argparse
import pandas as pd
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hotels Model Generator")
    parser.add_argument('--input_csv', required=True, help="Input CSV path to manipulate", type=str)
    parser.add_argument('--output_model_path', required=True, help="Where to export the created model", type=str)
    parser.add_argument('--train_size', required=False,
                        help="Percentage training size from the csv",
                        type=float,
                        default=0.8)
    parser.add_argument('--model_type', required=False,
                        help='Type of generated model can be either: DecisionTree, NaiveBayes, SVM',
                        choices=['DecisionTree', 'NaiveBayes', 'SVM'],
                        type=str,
                        default='DecisionTree')

    args = parser.parse_args(sys.argv[1:])

