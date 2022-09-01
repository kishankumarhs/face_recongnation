from genericpath import isfile
from itertools import count
import json
import face_recognition
from PIL import Image, ImageDraw
import os

# folder path
dir_path = r'dataset/'

# list to store files
res = []
known_face_encodings = []
known_face_names = {}
known_images = []
newImages = {

}
newImagesObject = {}
# Iterate directory
for path  in os.listdir(dir_path):
    newpathArray = []
    for subpath in  os.listdir(os.path.join(dir_path,path)):
        if os.path.isfile(os.path.join(dir_path,path,subpath)):
            newpathArray.append(subpath)
            newImages[path] = newpathArray
i = 0 
for path in os.listdir(dir_path):
    newImageArray= [] 
    for subpath in  os.listdir(os.path.join(dir_path,path)):
        if os.path.isfile(os.path.join(dir_path,path,subpath)):
            image = face_recognition.load_image_file("./dataset/"+path+"/"+subpath)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_images.append(image)
            newImageArray.append(i)
            known_face_names[path] = newImageArray
            i +=1

print(known_face_names)
print(known_face_encodings)
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("./test/Messi PSG Training.jpg")
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)


pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        for names in known_face_names.keys():
           if first_match_index in  known_face_names[names]:
             name = names
        known_image = known_images[first_match_index]
        show_data_image = Image.fromarray(known_image)
    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(48, 63, 159))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(48, 63, 159), outline=(48, 63, 159))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 0))



del draw

pil_image.show()
show_data_image.show()

