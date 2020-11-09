# Self-Driving-Car
Autonomous driving car's are the future of car's.It's quite complicated problem but i have tried to provide a solution for a part of the problem.
The Model predicts the steering angle frame by frame.


# DataSet Link:
The link to the dataset is https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view. The datset contains around 45k images and corresponding steering angles.

# Model insights
1. The steering angle was in degrees and it was converted into radians first.
2. The model was designed with basic CNN, Flatten, Dense and Dropout Layers.
3. The activation used in inner layers was relu
4. The activation used in output layer was tanh.
5. Dropout layers were added for regularization and to prevent overfitting.
6. The model predicted the value of steering angle in radians so later it was converted back to degrees.

