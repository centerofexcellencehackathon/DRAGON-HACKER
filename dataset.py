import pandas as pd
import cv2
import os
import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


############################################################################################

def Collect_face(name):
        def path_exists(path):
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
                return 0
            else:
                print("User exists....")
                count = file_count(path)
                return count
        
        def file_count(path):
            list = os.listdir(path) # dir is your directory path
            number_files = len(list)
            print("number of data set captured previouslh is :",number_files)
            return number_files
        
        vid_cam = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        folder=name
        count = 0
        path="Face_DataBase/"
        path+=(str(folder)+"/")
        count=path_exists(path)
        capture = False
        # Start looping
        print('press \'s\' to capture (for 100ms)')
        print('press \'p\' to pause (for 100ms)')
        print('press \'q\' to stop capturing and exist (for 100ms)')
        while(True):
        
            # Capture video frame
            ret, image_frame = vid_cam.read()
            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
 
                if cv2.waitKey(1)== ord('s'):
                    print("Started to capture")
                    capture = True
        
                elif cv2.waitKey(1)== ord('p'):
                    print("Stopped capturing")
                    capture = False
        
                if capture:
                    image_file_name=str(path + str(folder) + '_' + str(count) + ".jpg")
                    cv2.imwrite(image_file_name, image_frame[y:y+h,x:x+w])
                    print("File saved in path:  ",image_file_name)
                    count += 1
    
                cv2.imshow('frame', image_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("Images saved under ur Name!")
                print('Note: to recapture enter the same name!')
                print("Thanks! ",str(folder))
                break
            elif count>1000:
                
                break
        
        vid_cam.release()
        cv2.destroyAllWindows()
        train_face_to_model()
        
#####################################################################################
def train_face_to_model():
    print("Training!!!!")
    def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
                X = []
                y = []
            
                # Loop through each person in the training set
                for class_dir in os.listdir(train_dir):
                    if not os.path.isdir(os.path.join(train_dir, class_dir)):
                        continue
            
                    # Loop through each training image for the current person
                    for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                        image = face_recognition.load_image_file(img_path)
                        face_bounding_boxes = face_recognition.face_locations(image)
                        print("trainig for image : ",img_path)
            
                        if len(face_bounding_boxes) != 1:
                            # If there are no people (or too many people) in a training image, skip the image.
                            if verbose:
                                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                        else:
                            # Add face encoding for current image to the training set
                            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                            y.append(class_dir)
            
                # Determine how many neighbors to use for weighting in the KNN classifier
                if n_neighbors is None:
                    n_neighbors = int(round(math.sqrt(len(X))))
                    print("Chose n_neighbors automatically:", n_neighbors)
            
                # Create and train the KNN classifier
                knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
               
                knn_clf.fit(X, y)
            
                # Save the trained KNN classifier
                if model_save_path is not None:
                    with open(model_save_path, 'wb') as f:
                        pickle.dump(knn_clf, f)
            
                return knn_clf
            
    train_dir="Face_DataBase/"
    Training_flag=str(input("Do u want to train the model file? (y/n):  "))
    if(Training_flag == 'y'):
        
        print("Training KNN classifier...")
        classifier = train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")

########################################################################################
    
df=pd.read_csv("Database.csv")

print("Welcome to Igeeks portal")
nme=input("Enter the name: \n")
mail_Id=input("Enter your mail ID: \n")
while(1):
    pswd=input("Enter the pswd: \n")
    Re_pswd=input("Re-Enter the pswd: \n")
    if(pswd==Re_pswd):
        print("Registered")
        break
    else:
        print("Password doesnt match!! Re-Enter ")


name=[nme]
pswd=[pswd]
mail_Id=[mail_Id]

dict={"name":name,"mailid":mail_Id,"password":pswd}
df1=pd.DataFrame(dict)
print(df1)
df=df.append(df1)
df.to_csv("Database.csv",index=False)
#collect face Dataset
print("COLLECTING YOUR  DATASET of ur face!")
Collect_face(nme)


