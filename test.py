import json
import random
list_address = set()
# with open('/Users/trinhgiang/Downloads/address.json') as json_file:
#     data = json.load(json_file)
#     for p in data:
#         try:
#             raw_address = data[p]['raw_address'][0]
#             list_address.add(raw_address)
#         except:
#             continue
# import pandas as pd
# pair_sentence = []
# tag = []
# with open("/Users/trinhgiang/Downloads/check_acc.txt", 'r', encoding="utf-8", errors='ignore') as f:
#     sentences = f.readlines()
#     for sentence in sentences:
#         pair_sentence.append(sentence.strip())
#         tag.append(1)
#     for i in range(len(sentences)):
#         ran = random.randint(0, len(sentences) - 1)
#         tmp = sentences[ran].strip()
#         tmp2 = sentences[i].strip() + ' ' + tmp
#         pair_sentence.append(tmp2)
#         tag.append(0)
# with open("/Users/trinhgiang/Downloads/sentence_check_acc_2.txt", 'w+', encoding="utf-8", errors='ignore') as f:
#     for pair in pair_sentence:
#         f.write(pair)
#         f.write('\n')
# data = {'pair_sentence': pair_sentence, 'tag': tag}
# df = pd.DataFrame(data)
# df.to_csv("/Users/trinhgiang/Downloads/sentence_check_acc_2.csv", encoding='utf-8', index=True)

# from sklearn.metrics import accuracy_score
#
# df = pd.read_csv("/Users/trinhgiang/Downloads/sentence_check_acc.csv")
# result = df['tag']
# test = []
# for i in range(len(result)):
#     test.append(1)
#
# print(accuracy_score(result,test))
# ppls = []
# with open("/Users/trinhgiang/Downloads/result_ppl_2.txt", 'r', encoding="utf-8", errors='ignore') as f:
#     sentences = f.readlines()
#     for sentence in sentences:
#         ppls.append(float(sentence))
# for i in range(5,40):
#     count = 0
#     for j in range(0,670):
#         if (ppls[j]<i):
#             count+=1
#     print("Threshold = "+str(i)+" , "+"acc = "+str(count/670))

# import pickle
# with open('/Users/trinhgiang/Downloads/raw_data_split_sent', 'rb') as f:
#     docs =pickle.load(f)
#  str(docs[0]).replace()
# print(docs)

