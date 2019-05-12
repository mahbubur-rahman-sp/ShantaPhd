from androguard.misc import AnalyzeAPK
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis.analysis import ExternalMethod
import matplotlib.pyplot as plt
import networkx as nx
import re
import itertools
import jsonpickle

path = "c:\\tmp\\b.apk"
a, d, dx = AnalyzeAPK(path)


activities = a.get_activities()


# Note: If you create the CFG from many classes at the same time, the drawing
# will be a total mess...

activityList=[]

root = a.get_android_manifest_xml()
nsmap ="{"+ root.nsmap['android'] +"}"
for activity in root.iter('activity'):
    activityDetail = lambda: None
    activityDetail.name = activity.get(nsmap+'name').split('.')[-1]
    activityDetail.permission = activity.get(nsmap+'permission')
    
    activityDetail.intentFilters = []
    for intent in activity.findall('intent-filter'):
        try:
            intent_action = intent[0].get(nsmap+'name')
            activityDetail.intentFilters.append(intent_action)
        except:
            nnn=0    
    intents = []
    try:
        #a.get_intent_filters(
        for n in dx.find_classes(name='.*'+activityDetail.name, no_external=True):
            try:
                classInfo = n.get_vm_class()
                source = classInfo.get_source()
                x = re.findall("android\.intent\.\w+", source)
                intents.append(x)
                x = re.findall("android\.intent\.\w+\.\w+", source)
                intents.append(x)
               
                x = re.findall("Intent\(.+\)",source)
                for item in x:
                    try:
                        if '()' not in item:
                            intents.append([item.split("\"")[1]])
                    except:
                        rr=0    
                for act in a.get_activities():
                    actName = act.split('.')[-1]
                    x = re.findall('Intent.*this.*'+actName,source)
                    for item in x:
                        intents.append([item.split(" ")[-1]])
                    
            except:
                v = 9            
    except:
        n = 6              
              
    activityDetail.calledIntents = set(itertools.chain.from_iterable(intents)) - set(["android.intent.action"])
    activityList.append(activityDetail)
    print(jsonpickle.encode(activityDetail))
print(activityList)

