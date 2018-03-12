Step2
=====

Step 2 consists of classifying the manipulated data in step 1.

The classification is done using different classification algorithms:
- NaiveBayes (Gaussian)
- DecisionTree
- SVM
- OneVsAllClassifier Wrapper to each classifer above

The step consists of two executables
- HotelsPredictionCSVGenerator
    - Generates from the input csv,
      an output csv randomly that can be used as an input
      to the model to predict.
    - Usage:
      ```
      python HotelsPredictionCSVGenerator.py < ARGS >
      ```
      Args:
        - --input_csv < PATH >
        - --output_csv < PATH >
        - --generated_count < COUNT >
- HotelsModelPredictor
    - Used to both generate a model and use it to predict an input
    - On model generation, a validation on portion of the data is ran
      and stats are shown besides saving the model
    - Usage:
      ```
      python HotelsModelPredictor.py < ARGS > < MODE >
      ```
    - Args:
        - --input_csv < PATH >
        - mode [train, predict]
        - Train mode:
            - --output_model_folder < PATH >
            - --train_size < SIZE >
            - --enable_one_vs_rest
            - classifier [SVM, DecisionTree, NaiveBayes]
        - Test mode:
            - --input_model_folder < PATH >
            - --save_predictions_to_csv
    - Examples:
        - Train
          ```
          python HotelsModelPredictor.py --input_csv d.csv train --output_model_folder ./x --train_size 0.7 DecisionTree
          ```
        - Predict
          ```
          python HotelsModelPredictor.py --input_csv d.csv predict --input_model_folder ./x --save_predictions_to_csv
          ```