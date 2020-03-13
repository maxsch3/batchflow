Batch transformers

Batch transformers process incoming batch data as whole batch. The batch transformers have access to all
variables in data and they are for implementing all sorts of transforms involving any interaction between 
variables

# FeatureDropout

::: keras_batchflow.base.batch_transformers.FeatureDropout
    :docstring:

# ShuffleNoise

::: keras_batchflow.base.batch_transformers.ShuffleNoise
    :docstring:


# Base abstract classes

## BatchTransformer class

::: keras_batchflow.base.batch_transformers.BatchTransformer
    :docstring:

## Random cell base class
This class is a parent class for all transformers that transform random cells of incoming data using masked 
substitiution to some augmented version of the same batch. They all use the following formula at their core:

```python
batch = batch.mask(mask, augmented_version)
```

The child classes in general only differ by the way how augmented version is defined. For example:

1. FeatureDropout makes augmented version by simply creating a dataframe of the same structure as batch,
 but filled with drop values specified at initialization
2. Shuffle noise makes augmeted version by shuffling columns of batch  

::: keras_batchflow.base.batch_transformers.BaseRandomCellTransform
    :docstring:
    
#### Methods of random cell base class

::: keras_batchflow.base.batch_transformers.BaseRandomCellTransform
    :members: _make_mask _calculate_col_weights
