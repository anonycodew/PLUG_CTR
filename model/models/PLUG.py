import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BasicLayer import MultiLayerPerceptron, BasicCTRLLM
from model.loss_function import knowledge_alignment_loss

class PLUG_Framework(BasicCTRLLM):
    def __init__(self, ctr_model=None, plm_ctr="plm_ctr", projection_dim=32, llm_model=None,
                 llm_config = None, dropouts=(0.2,0.5), if_norm="no", tao=1.0, *args, **kwargs):
        super(PLUG_Framework, self).__init__(*args, **kwargs)
        self.ctr_model = ctr_model
        self.tao = tao
        self.dropout = nn.Dropout(dropouts[0])
        self.plm_ctr = plm_ctr
        self.if_norm = if_norm 
        

        self.text_feature_projection1 = MultiLayerPerceptron(llm_config.hidden_size,[llm_config.hidden_size//2], 
                                                            dropout=0.1, output_layer=False)
        
        if self.plm_ctr == "plm_ctr": 
            self.llm_output = MultiLayerPerceptron(llm_config.hidden_size//2, [256], 
                                                            dropout=0.1, output_layer=True)
        
        self.text_feature_projection2 = MultiLayerPerceptron(llm_config.hidden_size//2,[projection_dim], 
                                                            dropout=0.1, output_layer=False)
        
        assert self.ctr_model.tabular_feature_dimension, "The ctr_model should have feature_interaction_layers"
        self.tabular_feature_projection = MultiLayerPerceptron(self.ctr_model.tabular_feature_dimension, 
                                                               [self.ctr_model.tabular_feature_dimension//2, projection_dim],
                                                               dropout=0.1, output_layer=False)
        self.acti = torch.nn.ReLU()
    
    def compute_tabular_text_loss(self, tabular_features, semantic_features):
        if self.if_norm == "norm": 
            semantic_features = F.normalize(semantic_features, p=2, dim=-1)
        tabular_features = self.tabular_feature_projection(tabular_features)
        text_features = self.text_feature_projection2(self.text_feature_projection1(semantic_features))
        cl_loss = knowledge_alignment_loss(tabular_features, text_features, temperature=self.tao)
        return cl_loss        

    def ctr_prediction(self, user_item_ids, return_features=True):
        outputs = self.ctr_model(user_item_ids, return_features=return_features)
        return outputs
    
    def forward(self, user_item_ids, semantic_features, labels=None, return_cl_loss=True):
        outputs = self.ctr_prediction(user_item_ids, return_features=True)
        llm_outputs = None
        
        if self.plm_ctr == "plm_ctr": 
            llm_outputs = self.llm_output(self.text_feature_projection1(semantic_features))
            llm_outputs = self.output_activation(llm_outputs).squeeze(1)
            
        tabular_features = outputs["x_fi"] 
        y_pred = outputs["y_pred"] 

        cl_loss = self.compute_tabular_text_loss(tabular_features, semantic_features)
        if return_cl_loss:
            return_dict = {"y_pred": y_pred, "y_pred_llm":llm_outputs, "cl_loss":cl_loss}
        else:
            return_dict = {"y_pred": y_pred, "y_pred_llm":llm_outputs} 
        return return_dict
