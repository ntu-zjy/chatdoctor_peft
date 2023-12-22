# -*- coding: utf-8 -*-
# read the data json
import json
import sys
origin_data = "./data/chatdoctor5k_test.json"
method = sys.argv[1]
target_data = "./data/chatdoctor5k_eval_{}.json".format(method)
with open(origin_data, 'r') as file:
    origin_data_json = json.load(file)

with open(target_data, 'r') as file:
    target_data_json = json.load(file)

#print(origin_data_json)

"""###1. similarity based on SentenceTransformer"""
#!pip install sentence_transformers

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def cal_sim(target, origin):
  # Download model

  # The sentences we'd like to compute similarity about
  sentences = [target, origin]

  # Get embeddings of sentences
  embeddings = model.encode(sentences)

  # Compute similarities
  sim = util.cos_sim(embeddings[0], embeddings[1])
  #print("{0:.4f}".format(sim.tolist()[0][0]))
  return float("{0:.4f}".format(sim.tolist()[0][0]))

"""### 2.BLEU"""

#!pip install jieba
#!pip install nltk

import jieba
from nltk.translate.bleu_score import sentence_bleu

def cal_bleu(target, origin):
  target_fenci = ' '.join(jieba.cut(target))
  inference_fenci = ' '.join(jieba.cut(origin))


  # reference = [['this', 'is', 'a', 'duck']]
  reference = []
  candidate = []

  reference.append(target_fenci.split())
  candidate = (inference_fenci.split())
  score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
  score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
  score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
  score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
  reference.clear()

  return score1, score2, score3, score4

"""### ROUGE"""

#!pip install rouge

from rouge import Rouge

def cal_rouge(hypothesis, reference):
  rouger = Rouge()
  scores = rouger.get_scores(hypothesis, reference)

  return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']

"""###Calculate the Result"""

sim = []
score1, score2, score3, score4 = [], [], [], []
rouge1, rouge2, rougel = [], [], []
import statistics
from tqdm import tqdm

for target, origin in tqdm(zip(target_data_json, origin_data_json)):
  sim.append(cal_sim(target["output"],origin["output"]))
  s1, s2, s3, s4 = cal_bleu(target["output"],origin["output"])
  r1, r2, rl = cal_rouge(target["output"],origin["output"])
  score1.append(s1)
  score2.append(s2)
  score3.append(s3)
  score4.append(s4)
  rouge1.append(r1)
  rouge2.append(r2)
  rougel.append(rl)
print()
print("sim:",statistics.mean(sim))
print("score1:",statistics.mean(score1))
print("score2:",statistics.mean(score2))
print("score3:",statistics.mean(score3))
print("score4:",statistics.mean(score4))
print("Rouge1:",statistics.mean(rouge1))
print("Rouge2:",statistics.mean(rouge2))
print("Rougel:",statistics.mean(rougel))

# write the result into a txt file
with open("./results/chatdoctor5k_eval_{}.txt".format(method), 'w+') as f:
    f.write("sim: {}\n".format(statistics.mean(sim)))
    f.write("score1: {}\n".format(statistics.mean(score1)))
    f.write("score2: {}\n".format(statistics.mean(score2)))
    f.write("score3: {}\n".format(statistics.mean(score3)))
    f.write("score4: {}\n".format(statistics.mean(score4)))
    f.write("Rouge1: {}\n".format(statistics.mean(rouge1)))
    f.write("Rouge2: {}\n".format(statistics.mean(rouge2)))
    f.write("Rougel: {}\n".format(statistics.mean(rougel)))