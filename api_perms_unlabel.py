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
            permgroup={}
            apigroup={}
            permgroup["network"]=0
            permgroup["os"]=0
            permgroup["personal"]=0
            permgroup["system"]=0 
            permgroup["device"]=0
            apigroup["network"]=0
            apigroup["os"]=0
            apigroup["personal"]=0
            apigroup["system"]=0 
            apigroup["device"]=0
            for a in suspisus_permissions:
                found=0
                try:
                    a_name = a.split('|')[0]
                    a_grp = a.split('|')[1]
                    for i in apis:
                        if(a_name in i):
                            found=1
                            apigroup[a_grp]+=1
                            break
                    if(found==0):  
                        for i in perms:
                            if(a_name in i):
                                found=1
                                permgroup[a_grp]+=1
                                break  
                except:
                    f1 = 0                
                if(found==0):
                    f.write("0,")
                else:
                    f.write("1,")  
            f.write(str(permgroup["network"])+",")     
            f.write(str(permgroup["os"])+",")
            f.write(str(permgroup["personal"])+",")
            f.write(str(permgroup["system"])+",")
            f.write(str(permgroup["device"])+",")
            f.write(str(apigroup["network"])+",")     
            f.write(str(apigroup["os"])+",")
            f.write(str(apigroup["personal"])+",")
            f.write(str(apigroup["system"])+",")
            f.write(str(apigroup["device"]))
            
            f.write('\n')
        except Exception as error:
            print("Error in file "+filename+" "+str(error))   



suspisus_permissions = [line.strip(' \t\n\r') for line in open('dangerousPerms.txt')]
#print(suspisus_permissions)

f = open('dangerousPermUnlabel.csv', 'w')

for i in suspisus_permissions:
    f.write(i+",")

f.write("perm_network,perm_os,perm_personal,perm_system,perm_device,api_network,api_os,api_personal,api_system,api_device\n")


process_folder('C:\\dataset\\Unlabeled\\Androzo_Unlabeled_log',0)
process_folder('C:\\dataset\\Unlabeled\\apkmirror_pure_log',0)


#process_folder('C:\\Users\\ssharmee\\apk\\apkmirror_pure',0)
#process_folder('C:\\Users\\ssharmee\\apk\\unlabel',0)
 
#process_folder('/media/shanta/18EC3E31EC3E0990/linux/feature_vector_1',fileExtraction,0)
