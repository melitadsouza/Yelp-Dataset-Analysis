# dsouza{c, d, m}@indiana.edu
# Computes the accuracy of AFINN word-list sentiment analysis
import json
import re
from afinn import Afinn
from sklearn.metrics import confusion_matrix

SOURCE = 'D:\\YELP Dataset\\yelp_academic_dataset_review.json';

def normalize_text(text):
    text = re.sub('[^\x00-\x7F]+', ' ', text)
    return text

def getAFINNModelStats(count = None):
    """
    count ==> Analyses 'count' number of records only for computation, ignore if entire file needs to be scanned.
    """
    with open("D:\\YELP Dataset\\tobesubmitted\\restaurantID.txt", 'r') as busIds:
        id_list = busIds.read();
  
    c = 0  
    mismatch = 0
    y_true = []
    y_pred = []       
    afinn = Afinn();
    
    with open(SOURCE, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line) 
            if(data['business_id'] in id_list):
                data['text'] = ''.join([normalize_text(text) for text in data['text']]) 
                if(data['stars'] < 3):
                    senti_star = '-'
                elif(data['stars'] > 3):
                    senti_star = '+'
                else:
                    senti_star = 'N'
                
                a = afinn.score(data['text']);    
                if(a > 0):
                    senti_afinn = '+';
                elif(a < 0):
                    senti_afinn = '-';
                else:
                    senti_afinn = 'N'

                if(senti_star != senti_afinn):
                    mismatch = mismatch + 1;
                    
                y_true.append(senti_star);
                y_pred.append(senti_afinn);
                
                c = c + 1;
                if(count != None):
                    if(c == count):
                        break;
    
    print(confusion_matrix(y_true, y_pred, labels=["+", "-", "N"]))
    print("Count: {}\n".format(count))
    print("Mismatches: {}\n".format(mismatch))
    print("Accuracy: {} %\n".format((count - mismatch)/c * 100))
    print("Analysis completed.")

def main():
    getAFINNModelStats()
    
    """
    OUTPUT for complete .json file
    [[1592799   32759   23598]
     [ 294631  190506   35619]
     [ 330361   26925   13596]]
     
    Count: 2540794
    
    Mismatches: 743893
    
    Accuracy: 70.72 %
    
    Analysis completed.
    """
    
if(__name__ == "__main__"):
    main();