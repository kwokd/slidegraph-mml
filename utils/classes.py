from torch_geometric.data import Data

from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch

class MMLData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'omics_tensor': return None
        return super().__cat_dim__(key, value, *args, **kwargs)
    
class DFWrapper:
    # wrapper for the dataframe containing omics data (old)
    def __init__(self, surv_path, clini_path):
        # import DSS event-time data
        label = pd.read_excel(surv_path).rename(columns={'bcr_patient_barcode':'PATIENT'}).set_index('PATIENT')
        # filter to event/time columns for brca only
        label = label[["DSS","DSS.time"]][label.type == 'BRCA']
        
        ### import genomic data
        df = pd.read_excel(clini_path).set_index('PATIENT')
        # print(df.shape)
        
        # one-hot for each mutation value
        mut = pd.get_dummies(df.filter(regex="_mutation$"),dtype=float)

        # log1p all expression values
        expr = df.filter(regex="_expression$").apply(lambda x: [np.log1p(item) for item in x])

        # standard scale across all of the cnv values
        cnv = df.filter(regex="_CNV$|ZNF703")
        scaler = preprocessing.StandardScaler().fit(cnv)
        scaled = scaler.fit_transform(cnv)
        
        # join parts together
        df = pd.DataFrame(scaled, columns=cnv.columns, index=cnv.index)

        # due to inclusion of event-time data, should have 2 extra columns (97 vs 95)
        self.df = df.join(expr).join(mut).join(label,"PATIENT","inner").dropna()
        # print(self.df.shape)
        
    def get_tag_survlist(self, tags):
        # given a list of graph tags, attach the event/time only to those
        # that also exist in the dataframe (have omics), and add to a list
        tag_survlist = []
        for tag in tags:
            if tag in self.df.index:
                tag_survlist.append([tag, tuple(self.df.loc[tag,['DSS','DSS.time']])])
        return tag_survlist
    
    def get_tensor(self,tag): # get row at tag as a pytorch tensor
        row = self.df.loc[tag].drop(labels=["DSS","DSS.time"])
        return torch.tensor(row.values,dtype=torch.float)
    
    def get_omics_length(self): # remove 2 for the labels
        return self.df.shape[1] - 2
    
class SurvWrapper:
    def __init__(self, supp_path=r'./data/NIHMS978596-supplement-1.xlsx'):
        df = pd.read_excel(supp_path).rename(columns={'bcr_patient_barcode':'case_id'}).set_index('case_id')
        df = df[["DSS","DSS.time"]][df.type == 'BRCA']
        self.df = df.dropna()
    
    def get_tag_survlist(self, tags):
        # given a list of graph tags, attach the event/time only to those
        # that also exist in the dataframe (have omics), and add to a list
        return [[tag, tuple(self.df.loc[tag,['DSS','DSS.time']])] for tag in tags if tag in self.df.index]
    
class DFWrapperLarge:
    def __init__(self, csv_path=r'./data/omics/tcga_brca_omic.csv'):
        df = pd.read_csv(csv_path).set_index('case_id')

        # Filter rows based on column: 'slide_id'
        df = df[df['slide_id'].str.contains("DX1", regex=False, na=False)]

        # Drop columns: 'site', 'is_female' and 3 other columns
        df = df.drop(columns=['slide_id', 'site', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship', 'train'])
        
        self.df = df
        self.df_len = self.df.shape[1]

    def get_omics_length(self):
        return self.df_len    
    
    def filter_tags(self,tags):
        return [tag for tag in tags if tag in self.df.index]
    
    def get_tensor(self,tag): # get row at tag as a pytorch tensor
        return torch.tensor(self.df.loc[tag].values,dtype=torch.float)