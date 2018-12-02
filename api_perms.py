import os


def apiExtraction(path):
    with open(path) as f:
        content = list( filter(lambda x: 'api_call' in x,f.read().splitlines()))
        content = list(map(lambda x: x.split('::')[-1],content))
        content = list(set(content))
        return content

def featureExtraction(path):
    with open(path) as f:
        content = list( filter(lambda x: 'feature' in x,f.read().splitlines()))
        content = list(map(lambda x: x.split('::')[-1],content))
        content = list(set(content))
        return content

def permExtraction(path):
    with open(path) as f:
        content = list( filter(lambda x: 'permission' in x,f.read().splitlines()))
        content = list(map(lambda x: x.split('.')[-1],content))
        content = list(set(content))
        return content        

def process_folder(path,isMalware):
    for filename in os.listdir(path):
        try:
            filePath = path+"/"+filename
            print(filePath)
            #features = featureExtraction(filePath)
            apis = apiExtraction(filePath)
            perms = permExtraction(filePath)
            features = apis + perms

            for a in suspisus_permissions:
                found=0
                for i in features:
                    if(a in i):
                        found=1
                        break
                if(found==0):
                    f.write("0,")
                else:
                    f.write("1,")  
            f.write(str(isMalware))
            f.write('\n')
        except Exception as error:
            print("Error in file "+filename+" "+str(error))   



suspisus_permissions = [line.strip(' \t\n\r') for line in open('perm_call_feature.txt')]
#print(suspisus_permissions)

f = open('Unlabel.csv', 'w')

for i in suspisus_permissions:
    f.write(i+",")

f.write("IS_MALWARE\n")


# process_folder('C:\\Users\\ssharmee\\apk\\feature_vector_1',0)
# process_folder('C:\\Users\\ssharmee\\apk\\malware-feature-vector',1)
# process_folder('C:\\Users\\ssharmee\\apk\\feature_vector_2',0)
# process_folder("C:\\Users\\ssharmee\\apk\\txt4000",1)

process_folder('C:\\Users\\ssharmee\\apk\\apkmirror_pure',0)
process_folder('C:\\Users\\ssharmee\\apk\\unlabel',0)
 
#process_folder('/media/shanta/18EC3E31EC3E0990/linux/feature_vector_1',fileExtraction,0)
