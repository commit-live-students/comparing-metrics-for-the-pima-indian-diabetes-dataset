# Comparing Metrics for the Pima Indian Diabetes dataset

### The Setting

We have already encountered and used the Pima Indian Diabetes dataset a number of times in the classroom. Here is a nice function that will fetch the dataset and format it in the correct way for further processing:

        import numpy as np
        import pandas as pd

        def fetch_pid(url="https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"):
            names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
            dataframe = pd.read_csv(url, names=names)
            array = dataframe.values
            X = array[:,0:8]
            y = array[:,8]
            return((X, y))

        X, y = fetch_pid()

And here is an example of how we can build a simple Logistic Regression model with a 10-fold cross-validation, and then compute the f1 score:

        from sklearn import model_selection
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics

        def pid_lr(seed=42, scoring='f1'):
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            model = LogisticRegression()
            scoring = 'f1'
            results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

            print("f1 score: {:.3f} {:.3f}".format(results.mean(), results.std()))
            return(1)

        pid_lr()


### The Task

Your task is to write a function called `training_report()` that

1. Accepts the following parameters:
    - X, y (Numpy arrays for training; any format acceptable by sklearn will work)
    - seed (a number; a subsequent call to the function with the same seed will reproduce the same results)
    - KFold (the number of k-folds to be used in cross-validation)
    - classifier (a sklearn binary classifier)
2. Should return
    - A dataframe with 4 columns, 1 each for
        * `f1`
        * `accuracy`
        * `neg_log_loss`
        * `roc_auc`
    - The dataframe should have `KFlod` (which was provided as input) number of rows, containing the values of the respective metrics from cross-validated training
3. It should test the following requirements, and raise an exception otherwise:
    - seed can be coerced to a python integer
    - KFold can be coerced to a python integer


### Hint: Handling Exception (optional)

It is entirely possible to do this task by manually checking the types of input parameters, but this would be a good occassion to learn about `Exceptions` and see them in action. You can see how this is a useful trick in the final example of this section.

### Exceptions

1. What are exception?
    - errors detected during execution are called exceptions
2. [Read more about errors, exceptions, and handling them](https://docs.python.org/2/tutorial/errors.html)
3. When an exception happens, leading to an error, the type of exception can be from one of the [built-in exception types](https://docs.python.org/2/library/exceptions.html#bltin-exceptions). Some examples:
    - **ZeroDivisionError**: Raised when the second argument of a division or modulo operation is zero.
    - **NameError**: Raised when a local or global name is not found.
    - **TypeError**: Raised when an operation or function is applied to an object of inappropriate type.
    - **ValueError**: Raised when a built-in operation or function receives an argument that has the right type but an inappropriate value.
4. In the task above, we are concerned with `ValueError`

### Example

Here is an example of a function that accepts 2 parameters, `a` and `b`, and raises an exception when the following requirements are not met:

1. `a` is a python integer
2. `b` is a python integer


        def example(a, b):
            try:
                a = int(a)
            except ValueError:
                print("Please ensure the value of a is a Python integer.", "You entered: ", a)

            try:
                b = int(b)
            except ValueError:
                print("Please ensure the value of b is a Python integer.", "You entered: ", b)

            return(a+b)