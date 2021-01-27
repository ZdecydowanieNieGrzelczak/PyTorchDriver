from Digging.Test.Sample import Sample
import json
import pickle
import csv


sample = pickle.load(open("BestSample.p", "rb"))


with open("actions.csv", "w", newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
    wr.writerow(sample.actions)

with open("action_probs.csv", "w", newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
    wr.writerow(sample.action_probs)

with open('episode.json', 'w') as fp:
    json.dump(sample.episode_dict, fp)

