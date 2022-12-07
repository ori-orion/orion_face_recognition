gnome-terminal --command="roscore"

#--window-with-profile=HoldAfterFinish -e   

sleep 1 

gnome-terminal --window-with-profile=HoldAfterFinish --command="rosrun cv_camera cv_camera_node" --tab --title="as_findmatch" --command="roslaunch orion_face_recognition as_findmatch_flib_node.launch" --tab --title="action_client" --command="rosrun actionlib_tools axclient.py as_Findmatch orion_face_recognition/ActionServer_FindMatchAction"


