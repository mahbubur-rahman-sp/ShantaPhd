from androguard.misc import AnalyzeAPK
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis.analysis import ExternalMethod
import matplotlib.pyplot as plt
import networkx as nx
import re
import itertools

path = "c:\\tmp\\b.apk"
a, d, dx = AnalyzeAPK(path)


activities = a.get_activities()


# Note: If you create the CFG from many classes at the same time, the drawing
# will be a total mess...

intents = []
for act in a.get_activities():
    actName = act.split('.')[-1]
    try:
        #a.get_intent_filters(
        for n in dx.find_classes(name='.*'+actName, no_external=True):
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
              
intents = set(itertools.chain.from_iterable(intents)) - set(["android.intent.action"])
print(intents)

