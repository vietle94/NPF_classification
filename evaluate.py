import numpy as np
import pandas as pd
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--labels",help="csv file containing true labels",default="npf_train.csv")
parser.add_argument("--answers",help="csv file containing your answers",default="dummy.csv")
parser.add_argument("--accuracy",help="estimate of binary classificatin accuracy",default="0.5")
args = parser.parse_args()

labels = pd.read_csv(args.labels)
answers = pd.read_csv(args.answers)

n = labels.shape[0]

isevent_labels = np.array([1]*n)
isevent_labels[labels["event"]=="nonevent"] = 0

isevent_answers = np.array([1]*n)
isevent_answers[answers["event"]=="nonevent"] = 0

accuracy = np.mean(isevent_labels==isevent_answers)

accuracyofaccuracy = np.abs(float(args.accuracy)-accuracy)

perplexity = np.exp(-np.mean(np.log(np.where(isevent_labels==1,answers["p_event"],1-answers["p_event"]))))

multiaccuracy = np.mean(labels["event"]==answers["event"])

print("accuracy,perplexity,multiaccuracy,accuracyofaccuracy\n%.5f,%.5f,%.5f,%.5f" % (accuracy,perplexity,multiaccuracy,accuracyofaccuracy))





