import pandas as pd
import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

df=pd.read_csv("Database.csv")

Name=list(df["name"])
Mail_Id=list(df["mailid"])
Pswd=list(df["password"])
#####################################################################################################

def mail(msg,ID,attach=None):
    print("SENDING MAIL")
    print(msg)
    fromaddr = "raspberryp087@gmail.com"     #https://www.google.com/settings/security/lesssecureapps
    toaddr = ID
    msg_mail = MIMEMultipart()  
    msg_mail['From'] = fromaddr 
    msg_mail['To'] = toaddr
    if(attach!=None):
        print("with attachment")
        msg_mail['Subject'] = "Intruder found"
        body = msg
        msg_mail.attach(MIMEText(body, 'plain')) 
        filename = attach
        attachment = open(filename, "rb") 
        p = MIMEBase('application', 'octet-stream')  
        p.set_payload((attachment).read()) 
        encoders.encode_base64(p) 
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg_mail.attach(p)
        s = smtplib.SMTP('smtp.gmail.com', 587) 
        s.starttls() 
        s.login(fromaddr,"Raspberry@123") 
        text = msg_mail.as_string()
        s.sendmail(fromaddr, toaddr, text) 
        s.quit()
        print("MAIL SENT")
        return
    msg_mail['Subject'] = "IGEEKS SECURITY STATUS"
    body = msg
    msg_mail.attach(MIMEText(body, 'plain'))
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login(fromaddr,"Raspberry@123")  
    text = msg_mail.as_string()   
    s.sendmail(fromaddr, toaddr, text)   
    s.quit()
    print("MAIL SENT")
    return

######################################################
def Face_predict(Log_name,ID):
    def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
        
        if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(X_img_path))
    
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    
        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
    
        # Load image file and find face locations
        X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(X_img)
    
        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []
    
        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    
        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        
    
        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    
    
    def show_prediction_labels_on_image(frame, predictions):
        """
        Shows the face recognition results visually.
    
        :param img_path: path to image to be recognized
        :param predictions: results of the predict function
        :return:
        """
        
    
        for name, (top, right, bottom, left) in predictions:
            cv2.rectangle(frame,(left, top), (right, bottom),(0, 0, 255),2)
    
            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            #name = name.encode("UTF-8")
    
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
        # Remove the drawing library from memory as per the Pillow docs
    
        # Display the resulting image
        cv2.imshow('faceRecognisation',frame)
        
    print("Second Auth! in progress")
    print("Predicting!!!!  ")
    unknown_count=0
    know_count=0
    cap=cv2.VideoCapture(0)
    # STEP 2: Using the trained classifier, make predictions for unknown images
    while(1):
        ret,frame=cap.read()
        save_recent='recent.jpeg'
        cv2.imwrite(save_recent,frame)
        predictions = predict(save_recent, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print(" Found : {} ".format(name))
            if(name=='unknown' or name!= Log_name):
                unknown_count+=1
                if(unknown_count>5):
                    cv2.imwrite("unknown.jpg",frame)
                    msg="HI user\nSomeone tried to log in with ur password, we have attached the image of the intruder, please verify and change ur password "
                    print("UNKNOWN PERSON DTECTED")
                    mail(msg,ID,'unknown.jpg')
                    return
            elif name == Log_name:
                know_count+=1
                if(know_count>10):
                    unknown_count=0
                    print("LOGGED IN SUCCESFULL!!!!!!! ", name)
                    msg="HI user\n you logged in successful with IGEEKS SECURITY LOGIN"
                    mail(msg,ID)
                    return

                
            print("unknown_count:  ",unknown_count)
            print("Know_count:    ",know_count)

        # Display results overlaid on an image
        show_prediction_labels_on_image(frame, predictions)
        if(cv2.waitKey(1)==ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    pass


#######################################################
print("Welcome to IGeeks login PROTAL")
Ty_Name=input("Enter your Name:\n")

if Ty_Name in Name:
    
    index=Name.index(Ty_Name)
    Ty_Pswd=input("Enter your Password:\n")
    if Ty_Pswd == Pswd[index]:
        print("Correct password")
        Face_predict(Ty_Name,Mail_Id[index])
        

    else:
        print("Incorrect password")
        msg="Hi user\n Someone tried to log in with incorrect credentials"
        mail(msg,Mail_Id[index])
        
else:
    print("USER DOESNT EXSITS PLEASE REGISTER:")
