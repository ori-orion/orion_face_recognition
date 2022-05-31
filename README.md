# orion_face_recognition

## Package Prerequisites 

1. numpy>=1.14.0
3. pandas>=0.23.4
4. gdown>=3.10.1
5. tqdm>=4.30.0
6. Pillow>=5.2.0
7. opencv-python>=4.5.5.64
8. opencv-contrib-python>=4.3.0.36
9. tensorflow>=1.9.0
10. keras>=2.2.0
11. Flask>=1.1.2
12. mtcnn>=0.1.0
13. lightgbm>=2.3.1
14. dlib>=19.20.0
15. retina-face>=0.0.1
16. mediapipe>=0.8.7.3
17. fire>=0.4.0
18. fastai=='1.0.61'
19. pickle
20. rospy
21. actionlib
22. cv_bridge
23. glob
24. deepface


## Installation Guide
1. Git clone the repo
2. Download the file, 'ff_stage-1-256-rn50.pkl', in Google Drive, 'Robocup@Home->Perception->Facial Recognition', to 'orion_face_recognition/src/flib/'
3. (Suggested) Create a folder in /orion_face_recognition, called 'saved_faces'
4. build the package
5. (Suggested) Go through each configure file in 'orion_face_recognition/config/', change the parameter accordingly


## Capturing Face Action Server
This action server is used to register faces (potentially with different representations) to the database

**Action Name:** as_Capface

**Action msg path:** orion_face_recognition/ActionServer_CapFaceAction

**Action Specification: (type is in bracket)**
> Goal
> >(string) face_id 

> Result
> >(string) name
> >
> >(int32) If_saved


> Feedback
> >(float32) time

**Explanation:**

- face_id: specifying the name of the face. e.g, Tom, 1 ...

- name: same as face_id

- If_saved: indicating whether face is successfully saved in the database. (1 for success. 0 for failure)

- time: not assigned

**How to start the action server:**

1. Set the parameters in  
> orion_face_recognition/config/param_as_capface.yaml
2. Start the action server with command
> roslaunch orion_face_recognition as_capface_flib_node.launch


## Find Matched Face Action Server
This action server is used to compare the captured face from real-time camera to those saved in the database and return the most matched face in the database.
(second most matched face is also available)

**Action Name:** as_Findmatch

**Action msg path:** orion_face_recognition/ActionServer_FindMatchAction

**Action Specification: (type is in bracket)**
> Goal
> >(string) temp

> Result
>>(string) file_name
>>
>>(string) file_name_2ndbest
>>
>>(string) face_id
>>
>>(float32) best_match_score
>>
>>(float32) second_best_match_score


> Feedback
> >(float32) time

**Explanation:**

- temp: No need to assign any value, just call the action server

- file_name: the full path of the best matched face image file

- file_name_2ndbest: the full path of the second best matched face image file 

- face_id: the face_id of the best matched face image file

- best_match_score: for all captured face, the ratio of those matched to the best matched face image file in the total number of captured face
(will be combined with second_best_match_score if both files refer to the same person (but different representation))

- second_best_match_score: for all captured face, the ratio of those matched to the second best matched face image file in the total number of captured face

- time: not assigned


**How to start the action server:**

1. Set the parameters in  
> orion_face_recognition/config/param_as_findmatch.yaml
2. Start the action server with command
> roslaunch orion_face_recognition as_findmatch_flib_node.launch


## Find Attributes Action Server
This action server finds the attributes of input face image. The face image can either come from real-time camera or those saved in the database

**Action Name:** as_Findattrs

**Action msg path:** orion_face_recognition/ActionServer_FindAttrsAction

**Action Specification: (type is in bracket)**
> Goal
> >(string) face_id

> Result
>>(string[ ]) attrs
>>
>>(int32) num_attrs

> Feedback
> >(float32) time

**Explanation:**

- face_id: If leave empty, action server will capture faces from real-time camera output, then analyze the attributes of those captured faces. If a face_id is given, action server will analyze the attributes of all faces that match the face_id in the database

- attrs: a string list of attributes

- num_attrs: number of attributes in attrs

- time: not assigned


**How to start the action server:**

1. Set the parameters in  
> orion_face_recognition/config/param_as_findattrs.yaml
2. Start the action server with command
> roslaunch orion_face_recognition as_findattrs_flib_node.launch


## Clear Database Action Server
This action server clears all the files in the database

**Action Name:** as_Cleardatabase

**Action msg path:** orion_face_recognition/ActionServer_ClearDatabaseAction

**Action Specification: (type is in bracket)**
> Goal
> >(string) temp

> Result
>>(int32) Is_success

> Feedback
> >(float32) time

**Explanation:**
- temp: No need to assign any value, just call the action server
- Is_success: indicate whether the action is successful or not. (1 for success. 0 for failure)
- time: not assigned


**How to start the action server:**

1. Set the parameters in  
> orion_face_recognition/config/param_as_cleardatabase.yaml
2. Start the action server with command
> roslaunch orion_face_recognition as_cleardatabase_flib_node.launch



## Paremeter setting
### In 'param_as_general.yaml'

**Note: all those dir/path parameter must have '/' on both side**

**Relative path style: \~/orion_ws/src/orion_face_recognition/, tilde(~) is put at the start**

**Absolute path style: /home/jianeng/catkin_ws/src/orion_face_recognition/config, No tilde(~)**
- img_topic: image topic name to subscribe (e.g., /cv_camera/image_raw)
- database_dir: the directory that is used to save face images
- flib_path: directory of the flib class
- param_path: directoy to put all those yaml files

### In the other yaml files
- wait_time: Time for subscribing the camera topic
- max_face_num: Max number of faces (frames with face) saved in the '\_\_frame_list'
- num_representation: Number of representations for one face
- model_name: Pretrained model name for face recognition: VGG-Face, OpenFace, DeepFace, DeepID
- distance_metric: Metric name for face recognition: cosine, euclidean, euclidean_l2
- detector_backend: Detector backend for face detection: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
- normalization: Normalize the input image before feeding to model: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace



## Contact
If you have any question or find any bug, please contact: jianeng@robots.ox.ac.uk
