import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class RulesAssociation():
    def __init__(self, df: pd.DataFrame, columnsBind: list[str], binaryColumnName:str, parmColumn:str, valueParm: int ):
        
        """
            columnsBind - columns to bind
        """
        self.df = df.copy()        
        self.columnsBind = columnsBind
        self.binaryColumnName = binaryColumnName
        self.parmColumn = parmColumn
        self.valueParm = valueParm   
        
    def bind(self):
        for column in self.columnsBind:
            med = self.df[column].median()
            self.df[column] = (self.df[column] > med).astype(int)
    
    def getBinaryDF(self):
        return self.df[self.columnsBind + [self.binaryColumnName]].astype(bool)
 
    def applyApriori(self):
        df_bin = self.getBinaryDF()
        frequent_itemsets = apriori(df_bin, min_support=0.1, use_colnames=True)
        return frequent_itemsets
    
    def rules(self):
        frequent_itemsets = self.applyApriori()
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values(by='confidence', ascending=False)
        return rules