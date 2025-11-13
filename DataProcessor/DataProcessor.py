import pandas as pd

#Component
from DataProcessor.Processor import Processor

class DataProcessor:
    def __init__(self, csvPath: str, paramCleaner: str = None, columnsKeep: list[str] = None ):
        """
            Atributos
                csvPath         - caminho do data set
                paramCleaner    - coluna que queremos limpar
        """
              
        self.csvPath = csvPath
        self.paramCleaner = paramCleaner
        self.columnsKeep = columnsKeep
        self.df = None 
        
        self.processor = Processor(
            ['tempo','duration_ms'],
            [40,220,30000,600000], 
            df = self.df
        )
            
    def reader(self):
        self.df = pd.read_csv(self.csvPath)
        return self.df
        
    def keepColumns(self):
        if self.df is None:
            self.reader()
        
        if self.columnsKeep:
            self.df = self.df[[c for c in self.columnsKeep if c in self.df.columns]]
            return self.df
      
    def cleanCsv(self, outlier:bool):
        if self.df is None:
            self.reader()
        
        if self.paramCleaner and self.paramCleaner in self.df.columns:
            self.df.drop(columns=[self.paramCleaner], inplace=True)
            
        
        self.df.dropna(inplace = True) # Limpanos os Nan
        self.df.drop_duplicates(inplace = True) # limpamos os duplicado
        
        # Limpar com outliers
        if outlier:
            self.processor.df = self.df
            self.df = self.processor.outlierFilter()
        return self.df
    
    def normalizeCleanData(self, save = None):
        if self.df is None:
            raise ValueError("No DataFrame uploaded for normalization")
        
        self.processor.df = self.df
        
        df_normalized = self.processor.normalizeData()
        
        if save:
            df_normalized.to_csv("./Output/spotify_clean.csv", index=False)
            
        return df_normalized
    
    def debugger(self):
        if self.df is None:
            self.reader()
        
        print(self.df.shape)
        print(self.df.head())
        print(self.df.columns)
        print(self.df.info()) 

    