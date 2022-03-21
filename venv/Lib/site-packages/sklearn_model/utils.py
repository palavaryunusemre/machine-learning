import json
import math
from sklearn_model.export import Model

class JMLM(object):
    def __init__(self, model):
        self.jmlm = model

    @classmethod
    def fromFile(cls, filepath):
        model = None
        with open(filepath, "r") as jmlmfile:  
            model = json.load(jmlmfile) 
        return cls(model)

    @classmethod
    def fromString(cls, jmlm_string):
        model = json.loads(jmlm_string)
        return cls(model)

    @classmethod
    def fromModel(cls, model):
        if isinstance(model, Model):
            model = model.export()
        else:
            raise TypeError("model should be of type Model (sklearn_model.export.Model).")
        return cls(model)
        

    def extractRules(self, numericalToCategorical = False):
        if self.jmlm["model"]["type"] != "DecisionTreeClassifier":
            raise TypeError("Extract Rules only supports model of type 'DecisionTreeClassifier'.")

        categoricalMap = None
        if numericalToCategorical:
            categoricalMap = {}
            fields = [self.jmlm["input"], self.jmlm["output"]]
            for field in fields:
                for key, value in field.items():
                    if value["type"] == "category":
                        categoricalMap[key] = value["values"] 

        l = []
        self.traverseNode(self.jmlm["model"]["scoring_params"]["tree"], "", l, categoricalMap)
        return l 

    @staticmethod
    def traverseNode(node, rule_so_far, l, categoricalMap = None, ignoreMap = None):
        suffix = ""
        if node["isleaf"]:
            l.append({"class" : node["class"], "rule": rule_so_far})
            return

        variable = node["field"]
        split_val = node["split_value"]
        if len(rule_so_far) > 0:
            suffix = " and "
        left_rule = rule_so_far + suffix
        right_rule = rule_so_far + suffix
        ignoreMapLeft = None
        ignoreMapRight = None
        if categoricalMap and variable in categoricalMap:
            ignoreMapLeft = {k: v.copy() for k, v in ignoreMap.items()} if ignoreMap else {}
            ignoreMapRight = {k: v.copy() for k, v in ignoreMap.items()} if ignoreMap else {}
            split_idx = math.ceil(split_val)
            left_list = categoricalMap[variable][:split_idx]
            right_list = categoricalMap[variable][split_idx:]
            if ignoreMap and variable in ignoreMap:
                ignoreMapLeft[variable] = list(set(ignoreMap[variable] + right_list))
                ignoreMapRight[variable] = list(set(ignoreMap[variable] + left_list))
                left_list = [e for e in left_list if e not in ignoreMap[variable]]
                right_list = [e for e in right_list if e not in ignoreMap[variable]] 
            else:
                ignoreMapLeft[variable] = right_list.copy()
                ignoreMapRight[variable] = left_list.copy()

            if len(left_list) == 0 or len(right_list) == 0:
                raise ValueError("The decision for variable {0} with split value {1} cannot be empty.".format(variable, split_val))
            
            if len(left_list) == 1:
                left_rule += "({0} == '{1}')".format(variable, left_list[0])
            else:
                left_rule += "({0} in {1})".format(variable, left_list)
            
            if len(right_list) == 1:
                right_rule += "({0} == '{1}')".format(variable, right_list[0])
            else:
                right_rule += "({0} in {1})".format(variable, right_list)        
        else:
            left_rule += "({0} <= {1})".format(variable, split_val)
            right_rule += "({0} > {1})".format(variable, split_val)
        return (JMLM.traverseNode(node["l"], left_rule, l, categoricalMap, ignoreMapLeft), 
                JMLM.traverseNode(node["r"], right_rule, l, categoricalMap, ignoreMapRight))
