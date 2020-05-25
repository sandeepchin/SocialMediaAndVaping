# Social Media And Vaping

The goal of this work is to determine whether spending time on social media has an affect on vaping.

vaping.csv - contains survey data from more than 1100 high school students about social media usage and vaping habits.
SocialMediaAndVaping.ipynb is a Jupyter Notebook file that contains Python code to visualize and analyze the above data.
SocialMediaAndVaping.py is a standalone python file containing all the code from the above notebook.
SocialMediaAndVaping.html is a html version of above Jupyter Notebook.

The notebook goes through various steps of model-based learning:
- Preprocessing data.
- Data visualization for univariate and bivariate analysis.
  - involves converting categorical data into numeric data.
  - Social media usage is mapped into 6 groups numbered 0-5.
- Determine independent and dependent(target) variables.
  - Social media usage is independent variable(x).
  - Percentage of people who vape in each usage group is the target data (y).
- Split data into training and test sets.
- Build a model and train it.
- Test your model using test set.
- Predict probability of vaping for an unknown data point(usage group).
- Calculate metrics for evaluation of model.
- Draw the regression line - determine the slope and intercept of the line.
- Draw conclusions from the model.
- Calculate test statistic.



Conclusions:
- Coefficient of correlation shows a strong positive correlation between social media usage and vaping.

Ho (null hypothesis) : Social media has no influence on Vaping <br>
Ha (alternative hypothesis): Social media has influence on Vaping <br>

- test statistic > talpha/2, therefore we reject Ho. 
- Therefore, social media has influence over vaping. 

