
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import BasicCTRLLM, MultiLayerPerceptron


class LLMProbeCTR(BasicCTRLLM):
    def __init__(self, llm_config = None, dropouts=(0.2, 0.5),  if_norm="norm",*args, **kwargs):
        super(LLMProbeCTR, self).__init__(*args, **kwargs)
        

        self.text_feature_projection1 = MultiLayerPerceptron(llm_config.hidden_size,[llm_config.hidden_size//2], 
                                                    dropout=0.5, output_layer=False)
        
        self.llm_output = MultiLayerPerceptron(llm_config.hidden_size//2, [256], 
                                                        dropout=0.5, output_layer=True)
        self.if_norm = if_norm
    

    def ctr_model(self, semantic_features): 
        if self.if_norm == "norm":
            semantic_features = F.normalize(semantic_features, p=2, dim=-1)
        y_pred = self.llm_output(self.text_feature_projection1(semantic_features))
        return y_pred
    
    def forward(self, user_item_ids, semantic_features):
        y_pred = self.ctr_model(semantic_features)

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred,
                       "cl_loss": 0.0}
        return return_dict