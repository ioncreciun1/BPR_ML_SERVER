from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

class SignLanguageModel():
    def __init__(self):
        self.top_words =  np.array(["deaf","friend","thank you","hello","love"])
        self.model = Sequential()

        # input layer
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(60,150)))

        #hidden layers
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=False, activation='relu'))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))

        #output layer
        #Only 5 words are used
        self.model.add(Dense(5, activation='softmax'))

        # Loading a saved model
        self.model.load_weights("C:/Users/ion/Desktop/Personal/BPR_ML_SERVER/App/weights.h5")


    def extract_keypoints(self,pose_world_landmarks,right_hand_world_landmarks,left_hand_world_landmarks):
        pose = np.array([[round(res['x']), round(res['y'])] for res in pose_world_landmarks]).flatten() if pose_world_landmarks else np.zeros(12*2)
        lh = np.array([[round(res['x']), round(res['y']), round(res['z'])] for res in left_hand_world_landmarks ]).flatten() if left_hand_world_landmarks  else np.zeros(21*3)
        rh = np.array([[round(res['x']), round(res['y']), round(res['z'])] for res in right_hand_world_landmarks ]).flatten() if right_hand_world_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])


    def get_left_and_right_hand_landmarks(self, multi_hand_world_landmarks,multi_handedness):
        hands_position = multi_handedness
        hands_landmarks = multi_hand_world_landmarks


        if not hands_position:
            return [None, None]
        first_hand = hands_position[0]
        second_hand = None
        if(len(hands_position) == 2):
            second_hand = hands_position[1]
        
        if len(hands_position) == 1 or (second_hand and first_hand['label'] == second_hand['label']):
            if(len(hands_position) == 2 and first_hand['score'] < second_hand['score']):
                landmarks = hands_landmarks[1]
            else:
                landmarks = hands_landmarks[0]
            hand = []
            for item in landmarks:
                hand.append({'x': item['x'], 'y': item['y'], 'z': item['z']})

            if hands_position[0]['label'] == 'Left':
                return [None,hand]
            return [hand,None]
        
        left_hand = []
        right_hand = []
        left_hand_landmarks = hands_landmarks[0]
        right_hand_landmarks = hands_landmarks[1]
        for i in range(len(left_hand_landmarks)):
            right_hand.append({'x': left_hand_landmarks[i]['x'], 'y': left_hand_landmarks[i]['y'], 'z': left_hand_landmarks[i]['z']})
            left_hand.append({'x': right_hand_landmarks[i]['x'], 'y': right_hand_landmarks[i]['y'], 'z': right_hand_landmarks[i]['z']})

        return [left_hand, right_hand]


    def convert_landmarks_to_2d_array(self,pose,right_hand,left_hand):
        landmarks_as_array = []


        for i in range(len(pose)):    
            points = self.extract_keypoints(pose[i],right_hand[i],left_hand[i])
            landmarks_as_array.append(points)
        
        return landmarks_as_array


    def extract_landmarks_from_pose(self,pose):
        landmarks = pose.pose_landmarks
        pose_landmarks = []
        
        if(landmarks == None):
            return [None]
        else:
            landmarks = landmarks.landmark[11:23]
            for item in enumerate(landmarks):
                if(item==None):
                    pose_landmarks.append(None)
                else:
                    landmark = {'X':item.x,'Y':item.y,'Z':item.z}
                    pose_landmarks.append(landmark)
        return pose_landmarks

    def convert_to_60_frames(self,pose,right_hand,left_hand):
        pose_arr = pose
        right_hand_arr = right_hand
        left_hand_arr = left_hand

        if(len(right_hand_arr)<60):
                while len(right_hand_arr) < 60:
                    right_hand_arr.append(None)
                    pose_arr.append(None)
                    left_hand_arr.append(None)
        return [pose_arr,right_hand_arr,left_hand_arr]
        
    def get_preprocess_data(self, input_data):
        pose = input_data["POSE_LANDMARKS"]
        left_hands = []
        right_hands = []
        for i in range(len(input_data["HANDS_LANDMARKS"])):
            left_hand,right_hand = self.get_left_and_right_hand_landmarks(input_data["HANDS_LANDMARKS"][i],input_data["MULTI_HANDEDNESS"][i])
            left_hands.append(left_hand)
            right_hands.append(right_hand)
        pose,left_hands,right_hands = self.convert_to_60_frames(pose,left_hands,right_hands)
        return self.convert_landmarks_to_2d_array(pose,right_hands,left_hands)

    def predict(self, processed_data):
        res = self.model.predict(np.expand_dims(processed_data, axis=0))[0]
        return self.top_words[np.argmax(res)]