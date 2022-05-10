import torch
import numpy as np
import tensorflow as tf
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding
from textattack.attack_recipes import AttackRecipe
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker, AttackArgs


class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):
        text_array = np.array(text_input_list)
        preds = self.model(text_array).numpy()
        logits = torch.exp(-torch.tensor(preds))
        logits = 1 / (1 + logits)
        logits = logits.squeeze(dim=-1)
        final_preds = torch.stack((1-logits, logits), dim=1)
        return final_preds


model = tf.keras.models.load_model('ModelTestowy')


class MyAttack1(AttackRecipe):
    def build(model_wrapper):
        goal_function = UntargetedClassification(model_wrapper)
        constraints = [RepeatModification(),
                       StopwordModification(),
                       WordEmbeddingDistance(min_cos_sim=0.5, include_unknown_words=False),
                       UniversalSentenceEncoder(metric='cosine', window_size=20)]
        transformation = WordSwapEmbedding(max_candidates=20)
        search_method = GreedyWordSwapWIR(wir_method="weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)


model_wrapper = CustomTensorFlowModelWrapper(model)
dataset = HuggingFaceDataset("rotten_tomatoes", None, "validation")
attack = MyAttack1.build(model_wrapper)
attack_args = AttackArgs(shuffle=True, num_examples=100, checkpoint_dir='checkpoint')
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
