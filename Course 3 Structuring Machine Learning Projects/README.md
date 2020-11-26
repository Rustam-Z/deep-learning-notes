# [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)

- How to build a successful machine learning projects

- How to prioritize the problem

- ML strategy

- Choose a correct train/dev/test split of your dataset

- Human-level performance (Bayes error)

## Content
- Week 1
    - Introduction to ML strategy
        - [Why ML strategy](#Why-ML-strategy)

        - [Orthogonalization](#Orthogonalization)

    - Setting up your goal
        - [Single number evaluation metric](#Single-number-evaluation-metric)

        - [Satisficing and Optimizing metric](#Satisficing-and-Optimizing-metric)

        - [Train/dev/test distributions](#Train/dev/test-distributions)

        - [Size of the dev and test sets](#Size-of-the-dev-and-test-sets)

        - [When to change dev/test sets and metrics](#When-to-change-dev-test-sets-and-metrics)

    - Comparing to human-level performance
        - [Why human level performance](#Why-human-level-performance)

        - [Avoidable bias](#Avoidable-bias)

        - [Surpassing human-level performance](#Surpassing-human-level-performance)

        - [Improving your model performance](#Improving-your-model-performance) (summary)

    - [ML flight simulator](Bird_recognition_in_the_city_of_Peacetopia.pdf)

- Week 2

## Week 1: Introduction to ML strategy

### Why ML strategy
<img src="media/01.PNG" width=500>

### Orthogonalization
[Supplemental notes](Orthogonalization.pdf)

<img src="media/02.PNG" width=500>
<img src="media/03.PNG" width=500>

### Single number evaluation metric
[Supplemental notes](Single_number_evaluation_metric-2.pdf)

<img src="media/04.PNG" width=500>
<img src="media/05.PNG" width=500>

### Satisficing and Optimizing metric
[Supplemental notes](Satisficing_and_optimizing_metric.pdf)

### Train/dev/test distributions
[Supplemental notes](Training_development_and_test_distributions.pdf)

<img src="media/06.PNG" width=500>
<img src="media/07.PNG" width=500>
<img src="media/08.PNG" width=500>

- Setting up the `training`, `development` and `test` sets have a huge impact on productivity. It is important to choose the `development` and `test` sets from the **same distribution** and it must be taken randomly from all the data.

- Guideline: Choose a `development` set and `test` set to reflect data you expect to get in the future and consider
important to do well.

### Size of the dev and test sets
[Supplemental notes](Size_of_the_development_and_test_sets.pdf)

<img src="media/09.PNG" width=500>
<img src="media/10.PNG" width=500>
<img src="media/11.PNG" width=500>

### When to change dev/test sets and metrics
[Supplemental notes](When_to_change_develpment_test_sets_and_metrics.pdf)

### Why human level performance
[Supplemental notes](Why_human_level_performance.pdf)

### Avoidable bias
[Supplemental notes](Avoidable_bias.pdf)

### Understanding human-level performance
[Supplemental notes](Understanding_human_level_performance.pdf)

### Surpassing human-level performance
[Supplemental notes](Surpassing_human_level_performance.pdf)

### Improving your model performance
[Supplemental notes](Improving_your_model_performance.pdf)

<img src="media/12.PNG" width=500>
<img src="media/13.PNG" width=500>

