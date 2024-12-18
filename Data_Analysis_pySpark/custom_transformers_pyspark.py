from pyspark.ml import Transformer
import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark import keyword_only

spark = SparkSession.builder.appName("custom transformer").getOrCreate()

# create test dataframe
test_df = spark.createDataFrame(
    [[1, 2, 4, 1], [3, 6, 5, 4], [9, 4, None, 9], [11, 17, None, 3]],
    ["one", "two", "three", "four"]
)

# starting point for custom transformer by sub-classing Transformer
class ScalarNAFiller(Transformer):
    pass

# create a function to check transformer with
def scalarNAFillerfunction(
        df: DataFrame, inputCol: Column, outputCol: str,
        filler: float = 0.0):
    return df.withColumn(outputCol, inputCol).fillna(
        filler, subset=outputCol)

scalarNAFillerfunction(test_df, F.col("three"), "five", -99.0).show()

# create params for the filler using Param class
filler = Param(
    parent=Params._dummy(),
    name="filler",
    doc="value we want t replace our null with",
    typeConverter=TypeConverters.toFloat
)


# combine common used params in a Mixin with param.shared module
class HasInputCols(Params):
    """Mixin for param inputCols: input column names"""
    
    inputCols = Param(
        Params._dummy(),
        "inputCols",
        "input column name",
        typeConverter=TypeConverters().toListString
    )
    
    def __init__(self):
        super(HasInputCols, self).__init__()
        
    def getInputCols(self):
        return self.getOrDefault(self.inputCols)
    
# the scalarNAFiller with the params defined
class ScalarNAFiller(Transformer, HasInputCol, HasOutputCol):
    
    filler = Param(
        Params._dummy(),
        "filler",
        "value to replace null with",
        typeConverter=TypeConverters.toFloat
        )
    
    pass


# define the setters
"""
Blueprint: 
    from pyspark import keyword_only
    
    @keyword_only
    def setParams(self, *, inputCol=None, outputCol=None, filler=None):
        kwargs = self.input_kwargs
        return self._set(**kwargs)
"""

# individual setters
def setFiller(self, new_filler):
    return self.setParams(filler=new_filler)

def setInputCol(self, new_inputCol):
    return self.setParams(inputCol=new_inputCol)

def setOutputCol(self, new_outputCol):
    return self.setParams(outputCol=new_outputCol)

# define the getters
"""
Blueprint:
    def getFiller(self):
        return self.getOrDefault(self.filler)
"""

# the scalarNAFiller with setters and getters
class ScalarNAFiller(Transformer, HasInputCol, HasOutputCol):
    
    filler = Param(
        Params._dummy(),
        "filler",
        "value to replace null with",
        typeConverter=TypeConverters.toFloat
        )
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, filler=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setFiller(self, new_filler):
        return self.setParams(filler=new_filler)
    
    def getFiller(self):
        return self.getOrDefault(self.filler)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)
    
    
#%% the source code for the scalarNAFiller
class ScalarNAFiller(Transformer, HasInputCol, HasOutputCol):
    
    filler = Param(
        Params._dummy(),
        "filler",
        "value to replace null with",
        typeConverter=TypeConverters.toFloat
        )
    
    # definition of the initializer function
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, filler=None):
        super().__init__()
        self._setDefault(filler=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, filler=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setFiller(self, new_filler):
        return self.setParams(filler=new_filler)
    
    def getFiller(self):
        return self.getOrDefault(self.filler)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)
        
    # definition of the calculation function
    def _transform(self, dataset):
        if not self.isSet("inputCol"):
            raise ValueError(
                "No input column set for the" \
                    "ScalarNAFiller transformer."
            )
        input_column = dataset[self.getInputCol()]
        output_column = self.getOutputCol()
        na_filler = self.getFiller()
        
        return dataset.withColumn(
            output_column, input_column.cast("double")
            ).fillna(na_filler, output_column)


#%% USING OUR TRANSFORMER
test_ScalarNAFiller = ScalarNAFiller(
    inputCol="three", outputCol="five", filler=-99.9)

test_ScalarNAFiller.transform(test_df).show()


#%% TESTING CHANGES TO THE TRANSFORMER
test_ScalarNAFiller.setFiller(17).transform(test_df).show()

test_ScalarNAFiller.transform(
    test_df, params={test_ScalarNAFiller.filler: 17}
).show()


#%% CREATING A CUSTOM ESTIMATOR CLASS

# p344