import os
import feature



def process_folder(path):
    for filename in os.listdir(path):
        try:
            filePath = path+"\\"+filename
            f = open('C:\\Users\\ssharmee\\apk\\unlabel\\'+filename+'.txt', 'w')
            perms = feature.get_permissions(filePath)
            for p in perms:
                f.write('\npermission::'+p)
            apis = feature.get_apis(filePath)
            for a in apis:
                f.write('\napi_call::'+a.split('(')[0].replace('Landroid','android'))
        except Exception as error:
            print("Error in file "+filename+" "+str(error))   



#process_folder('c:\\linux\\apkmirror\\apks')

process_folder('C:\\Users\\ssharmee\\apk\\v4000')
 
#process_folder('/media/shanta/18EC3E31EC3E0990/linux/feature_vector_1',fileExtraction,0)
