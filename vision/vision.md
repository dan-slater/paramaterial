# Vision

The vision of this project is to provide a software tool that gets adapted by the material science and mechanical enginerering community. Presently, analyses from material test data are published without showing all the processing steps moving from raw measurements through to the final published results.

If instead, the approach proposed in this project is used, results would be published with the code that was used during processing. By referencing the version of paramaterial that was used to do the analyses, the results are made repeatable. 

The goal is to improve the quality and quantity of data available for materials modeling and simulation. It is hoped that using Paramaterial will help improve repeatability and reproducibility of materials test data analyses. This is achieved through automation and documentaion. Automation is achieved through the use of the Python programming language and the Paramaterial package. Documentation is achieved through the use of Jupyter notebooks and the Paramaterial package. The package was designed to help improve the quality and quantity of data available for materials modeling and simulation. It is hoped that using Paramaterial will help improve repeatability and reproducibility of materials test data analyses, and help to reduce the time and effort required to perform such analyses.

## Example

The following example shows how to use Paramaterial to perform a simple materials test data analysis. The example uses a dataset of tensile test data for a steel material. The dataset is available in the `paramaterial` package, and can be loaded using the `load_dataset` function. The dataset contains the following columns:

* `strain`: The strain of the material.
* `stress`: The stress of the material.

### Import packages

The first step is to import the required packages.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import paramaterial
from paramaterial import load_dataset
```

### Load dataset

The dataset is loaded using the `load_dataset` function. The dataset is returned as a `pandas` dataframe.

```python
df = load_dataset("steel_tensile")
```

### Plot dataset

The dataset is plotted using the `seaborn` package.

```python
sns.lineplot(data=df, x="strain", y="stress")
plt.show()
```

![png](docs/img/README.md-plot_dataset.png)

### Fit model

The dataset is fitted using the `fit_model` function. The `fit_model` function returns a `ModelResults` object, which contains the fitted model parameters, residuals, and other model results.

```python
model = paramaterial.fit_model(
    df=df,
    model="linear",
)
```

### Plot model

The model is plotted using the `seaborn` package. The fitted model is plotted using the `ModelResults` object returned by the `fit_model` function.

```python
sns.lineplot(data=df, x="strain", y="stress")
sns.lineplot(data=model.df, x="strain", y="stress")
plt.show()
```

![png](docs/img/README.md-plot_model.png)

### Plot model residuals

The model residuals are plotted using the `seaborn` package. The residuals are calculated using the `ModelResults` object returned by the `fit_model` function.

```python
sns.lineplot(data=model.df, x="strain", y="residual")
plt.show()
```

![png](docs/img/README.md-plot_model_residuals.png)

### Print model results

The model results are printed using the `print_model_results` function. The model results are printed using the `ModelResults` object returned by the


While the project is still in development, it is hoped that within a few years it will be used by the material science and mechanical engineering community. 

