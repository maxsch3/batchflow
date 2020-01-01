# Overview

The framework has layered and modular architecture. Each instance of batch generator is actually
a stack of 3 layers of functionality

- **Batch generator** - it samples batches from a full dataset. The batches sampled are
 pandas dataframes of the same structure as full dataset. A generator passes batch to
  the next layer 
- **Batch transformers** - makes transformations to a sampled batch. It has access to all variables
 in multi variable scenario and therefore can be used for transformations where variables
  interact. E.g. feature dropout where you have number of features per each data point and 
  you would like to drop one of them randomly. You can specify multiple transformers, which will 
  be applied sequentially
- **Sklearn transformers** - These are normally encoders that encode your data into keras friendly
 format. In the structure definition you specify which sklearn transformer you would like to
 be applied to which column of the dataset dataframe

At each level, there is a choice of interchangeable types of objects that you can use 
making a batch generator with a functionality you need. On the top of that you can create 
custom types of layers making the framework very flexible and extendable

# Components 

The framework comes with few standard components that you can choose from and combine together to 
make a generator with a required functionality. 

Below sections describe those out of the box components

## Batch generators

Batch generators 

## Batch transformers

