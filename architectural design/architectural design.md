# Architectural design

The following sections describe the architectural design of Paramaterial.

## Overview

The `DataSet` and `DataItem` classes are the core of Paramaterial. The `DataSet` class represents a collection of `DataItem` objects. The `DataItem` class represents a single data item, such as a single material test. The `DataSet` class provides methods for loading and saving data, and for performing operations on the data. The `DataItem` class provides methods for performing operations on a single data item.

Functions in the `paramaterial` package are used to perform operations on `DataSet` and `DataItem` objects. These functions are used to perform operations like cleaning data, performing calculations, and plotting data. 

The 'ModelSet' and 'ModelItem' classes are used to model the 'DataSet' and 'DataItem' classes. The 'ModelSet' class represents a collection of 'ModelItem' objects.

## Modus operandi

Starting from a dataset of raw test measurements, Paramaterial can be used to perform the following steps:

1. Preparation
2. Processing
3. Modelling
4. Analysis

### Preparation

The preparation step involves preparing the dataset for batch processing. This includes:

* Cleaning the data
* Standardising the data
* Checking for duplicates
* Exploring the data
* Creating a metadata table

### Processing

The processing step involves applying corrections to the data so that it can be used for modelling. This includes:

* 
