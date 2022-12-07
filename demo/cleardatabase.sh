gnome-terminal --command="roscore"

#--window-with-profile=HoldAfterFinish -e   

sleep 1 

gnome-terminal --window-with-profile=HoldAfterFinish --command="roslaunch orion_face_recognition as_cleardatabase_flib_node.launch" --tab --title="action_client" --command="rosrun actionlib_tools axclient.py as_Cleardatabase orion_face_recognition/ActionServer_ClearDatabaseAction"

