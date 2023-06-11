# Student_predict_ML

This code using `Machine Learning` libraries such as `Linear Regression, RandomForestRegressor, Pipeline, etc` and `Lazypredict(LazyRegressor)` predicts the math scores of students based on their reading and writing scores, parental level of education, gender, lunch, and test preparation course.

<h3> Requirements </h3>

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn

<h3> Usage </h3>
<p>To run the code, first install the required libraries, for example:</p>

```pip install sklearn ```

Then, run the following command:

```python students.py ```

This will train a model on the data in the `StudentScore.xls` file and then predict the math scores of the students in the test set. The results will be display in the terminal of your IDE. 

<h3> Limitations </h3>

- The code only considers a limited set of features. Other factors, such as socioeconomic status and home environment, may also affect student math scores.
- The code is not yet tested on a large dataset.
