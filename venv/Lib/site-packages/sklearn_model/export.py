import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

class Model(object):
    def __init__(self):
        self._dfX = None
        self._dfY = None
        self._model = None
        self._transformer = None
        self._transformer_fields = None
        self._output = None
        self._leaf_class = None

    def add_fields(self, dfX, dfY = pd.DataFrame()):
        self._dfX = None
        self._dfY = None
        if isinstance(dfX, pd.DataFrame):
            self._dfX = dfX
        else:
            raise TypeError("dfX should be a pandas DataFrame.")
        
        if isinstance(dfY, pd.DataFrame):
            if not dfY.empty:
                self._dfY = dfY
        else:
            raise TypeError("dfY should be a pandas DataFrame.")

    def add_transformer(self, transformer, transformer_fields):
        self._transformer = None
        self._transformer_fields = None
        if not isinstance(transformer_fields, (list, tuple)):
            raise TypeError("transformer_fields should be of type list or tuple (currently: {0})\
                ".format(type(transformer_fields)))
        if isinstance(transformer, (StandardScaler, MinMaxScaler)):
            if len(transformer_fields) != len(transformer.scale_):
                raise ValueError("Length of transformer_fields ({0}) does not match the number of columns\
                     in the transformer ({1}).".format(len(transformer_fields), 
                                                    len(transformer.scale_)))
            else:
                self._transformer = transformer
                self._transformer_fields = transformer_fields
        else:
            raise NotImplementedError("StandardScaler and MinMaxScaler are only supported transformers.")

    def add_model(self, model):
        self._model = None
        supportedTypes = (LinearRegression, DecisionTreeClassifier, KMeans)
        if isinstance(model, supportedTypes):
            self._model = model
        else:
            raise NotImplementedError("LinearRegression, DecisionTreeClassifier & KMeans models are currently supported.")
    
    def add_output_field(self, out_name, out_type, values=[]):
        self._output = None
        validTypes = ("bool", "int", "float", "category")
        if out_type in validTypes:
            output = {}
            output[out_name] = {}
            output[out_name]["type"] = out_type
            if out_type == "category":
                if isinstance(values, (list, tuple)):
                    if len(values) > 0:
                        output[out_name]["values"] = list(values)
                    else:
                        raise ValueError("Please provide a non-empty list or tuple as 'values' for categorical type output.")
                else:
                    raise TypeError("Parameter values can be a list or tuple.")
            self._output = output
        else:
            raise ValueError("Parameter type can only have one of the following values - {0}.".format(', '.join(validTypes)))
    
    def add_leaf_class(self, leaf_class):
        self._leaf_class = None
        if isinstance(leaf_class, dict):
            self._leaf_class = leaf_class
        else:
            raise TypeError("leaf_class should be of type dict.")

    def export(self):
        model_dict = {}
        if self._dfX is not None:
            model_dict["input"] = generate_fields(self._dfX)
        else:
            raise ValueError("_dfX cannot be None. Did you run add_fields() method?")

        if self._dfY is not None:
            model_dict["output"] = generate_fields(self._dfY)
        elif self._output is not None:
            model_dict["output"] = self._output
        else:
            raise ValueError("output field cannot be empty. Please provide it via dfY \
                parameter of add_fields() method or you can also add it using add_output_field() method?")

        if self._transformer is not None:
            model_dict["transformer"] = generate_transformer(self._transformer_fields, self._transformer)
        
        if self._model is not None:
            model_dict["model"] = generate_model(self._model, list(self._dfX.columns), self._leaf_class)
        return model_dict

    def exportJSON(self, filepath = None):
        model_dict = self.export()
        if filepath is not None:
            with open(filepath, "w") as outfile:  
                json.dump(model_dict, outfile, indent = 4) 
        else:
            model_json = json.dumps(model_dict, indent = 4)   
            return model_json

def generate_fields(df):
    field_dict = {}
    type_map = {"b": "bool",
                "i": "int",
                "u": "int",
                "f": "float",
                "O": "category"}
    
    for field in df.dtypes.index:
        field_dict[field] = {}
        if df.dtypes[field].kind in type_map:
            if df.dtypes[field].kind == "O" and df.dtypes[field].name != "category":
                raise TypeError("Object field {0} can only be of type category".format(field, df.dtypes[field].name))
            field_dict[field]["type"] =  type_map[df.dtypes[field].kind]
            if df.dtypes[field].name == "category":
                field_dict[field]["values"] = df.dtypes[field].categories.to_list()
        else:
            raise TypeError("Unsupported field {0} of type {1}".format(field, df.dtypes[field].name))
    return field_dict

def generate_transformer(fields, transformer):
    if isinstance(transformer, StandardScaler): 
        mean = transformer.mean_.tolist()
        stddev = transformer.scale_.tolist()     
        td = { fields[idx]: {"mean": mean[idx], "stddev": stddev[idx]} for idx in range(len(fields)) }
        return {"type": "Standard", "scale_fields": td}
    if isinstance(transformer, MinMaxScaler):
        scale = transformer.scale_.tolist()
        min_ = transformer.min_.tolist()
        td = { fields[idx]: {"scale": scale[idx], "min": min_[idx]} for idx in range(len(fields)) }
        return {"type": "MinMax", "scale_fields": td}
    else:
        raise NotImplementedError("StandardScaler and MinMaxScaler are only supported transformers.")

def generate_model(model, inputFields, leaf_class):
    if isinstance(model, LinearRegression):
        return generate_linregr_model(model, inputFields)
    if isinstance(model, DecisionTreeClassifier):
        return generate_decisiontree_model(model, inputFields, leaf_class)
    if isinstance(model, KMeans):
        return generate_kmeans_model(model, inputFields)
    else:
        raise NotImplementedError("LinearRegression, DecisionTreeClassifier & KMeans models are currently supported.")

def generate_linregr_model(model, inputFields):
    coeff = model.coef_.tolist()
    intercept = model.intercept_.tolist()     
    cd = { inputFields[idx]: coeff[idx] for idx in range(len(inputFields)) }
    return {"type": "LinearRegression", "scoring_params": {"coefficients": cd, "intercept": intercept}}

def generate_decisiontree_model(model, inputFields, leaf_class):
    left      = model.tree_.children_left
    right     = model.tree_.children_right
    threshold = model.tree_.threshold
    features  = [inputFields[i] for i in model.tree_.feature]
    
    def treeWalk(nodeId):
        if nodeId in leaf_class.keys():
            return { "isleaf" : True,
                    "class" : leaf_class[nodeId]}
        else:
            return {
                "isleaf": False,
                "field": features[nodeId],
                "split_value": threshold[nodeId] ,
                "l": treeWalk(left[nodeId]),
                "r": treeWalk(right[nodeId])
            }
    
    return {"type": "DecisionTreeClassifier", "scoring_params": {"tree": treeWalk(0)}}

def generate_kmeans_model(model, inputFields):
    l = []
    for center in model.cluster_centers_:
        l.append(dict(zip(inputFields, center)))
    return {"type": "KMeans", "scoring_params": {"metric": "euclidean", "centers":l }}

