# -*- coding: utf-8 -*-


import os
import warnings

import h5py
import pandas as pd

print(h5py.__version__)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
VERSION = 4
RANDOM_SEED = 7
import librosa
import librosa.display
from feature_extraction_utils import *

# from IPython.display import Audio

from tensorflow.keras.models import load_model

"""## Load data - To Define your selected features
## change to be aligned with your model input!
"""


df_joint_train_aug  = pd.read_csv('feature_name_load.csv', low_memory=False)
feature_column_names = [i for i in df_joint_train_aug.columns \
                        if i not in ['file_path','renamed_file_path','split','sentiment_value','emotional_category']]
#
def generate_selected_features_by_type(feature_column_names, input, stats, number=1):
    selected_result = []
    for name in feature_column_names:
        if input + "_" + stats in name:
            selected_result.append(name)
    if number < len(selected_result):
        selected_result = selected_result[:number]
    return selected_result


# example to take mfcc 20 mean & std; mel32; zcr all 5 stats features
feature_MFCC20_mean = generate_selected_features_by_type(feature_column_names, "mfcc", "mean", 20)
feature_MFCC20_std = generate_selected_features_by_type(feature_column_names, "mfcc", "std", 20)
feature_mel32_median = generate_selected_features_by_type(feature_column_names, "mel32", "median", 32)
feature_mel32_std = generate_selected_features_by_type(feature_column_names, "mel32", "std", 32)
feature_zcr_stats = generate_selected_features_by_type(feature_column_names, "zcr", "", 5)
feature_rms_stats = generate_selected_features_by_type(feature_column_names, "rms", "", 5)
selected_spect = ['Spectrum_band_energy_difference', 'Spectrum_band_density_difference',
                  'Spectrum_center_of_gravity_spectrum', 'Spectrum_skewness_spectrum', 'Spectrum_kurtosis_spectrum',
                  'Spectrum_stddev_spectrum', 'Spectrum_band_density', 'Spectrum_band_energy']
selected_formant = ['Formant_f1_mean', 'Formant_f1_median', 'Formant_f3_mean', 'Formant_fitch_vtl', 'Formant_mff',
                    'Formant_formant_dispersion']
selected_pitch = ['Pitch_pitch_slope_without_octave_jumps', 'Pitch_q3_pitch', 'Pitch_stddev_pitch',
                  'Pitch_mean_absolute_pitch_slope', 'Pitch_mean_pitch', 'Pitch_max_pitch', 'Pitch_q1_pitch',
                  'Pitch_min_pitch']
selected_intensity = ['Intensity_max_intensity', 'Intensity_q3_intensity', 'Intensity_median_intensity',
                      'Intensity_mean_intensity', 'Intensity_stddev_intensity', 'Intensity_relative_max_intensity_time']
selected_HNR = ['HNR_stddev_hnr', 'HNR_mean_hnr', 'HNR_relative_min_hnr_time', 'HNR_max_hnr']
selected_prosody = selected_intensity + selected_pitch  # + ['Local Jitter','Local Shimmer']
selected_feature_names131 = feature_MFCC20_mean + feature_MFCC20_std + feature_mel32_median + feature_mel32_std + \
                            feature_rms_stats + selected_intensity + selected_pitch + selected_spect

selected_feature_names128 = feature_MFCC20_mean + feature_MFCC20_std + feature_mel32_median + feature_mel32_std + \
                            feature_zcr_stats + feature_rms_stats + selected_intensity + selected_pitch

selected_feature_name = selected_feature_names128
len(selected_feature_name)
### define the selected feature names same as trained model!!!

print(selected_feature_name)

"""### TODO Load Model """
NCS_SEN_CNN_MODEL = load_model("./models/NCS_SEN_CNN_T2_S1S3S2Aa_0916-B-7805.h5", compile=False)
EMO_CNN_MODEL = load_model("./models/T2-0329-aug-VAL-R10-6042.h5", compile=False)
NCA_LAN_MLP_MODEL = load_model("./models/NCS_LAN_MLP_V2_0916-A2-9722.h5", compile=False)

# region load Data
"""### Load Data - test input from wav file"""


def get_stats_from_feature(feature_input):
    feature_mean, feature_median = np.mean(feature_input.T, axis=0), np.median(feature_input.T, axis=0)
    feature_std = np.std(feature_input.T, axis=0)
    feature_p10, feature_p90 = np.percentile(feature_input.T, 10, axis=0), np.percentile(feature_input.T, 90, axis=0)
    return np.concatenate((feature_mean, feature_median, feature_std, feature_p10, feature_p90), axis=0)


def calc_feature_all(filename):
    sample_rate_set = 16000
    X_full, sample_rate = librosa.load(filename, sr=sample_rate_set)

    # 获取音频的实际时长
    audio_duration = librosa.get_duration(y=X_full, sr=sample_rate)
    print(f"Audio duration for {filename}: {audio_duration:.2f} seconds")

    # 丢弃小于 0.128 秒的音频文件
    if audio_duration < 0.128:
        print(f"Skipping file {filename} because it is too short (<0.128s).")
        return

    # 如果音频小于 0.2 秒，设置 duration_to_use 为 0.2 秒，否则使用实际时长
    duration_to_use = max(audio_duration, 0.2)

    # 加载音频文件，使用调整后的时长
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast', duration=duration_to_use, sr=sample_rate_set,
                                  offset=0)

    # 检查音频是否为空
    if len(X) == 0:
        print(f"Skipping file {filename} because it is empty.")
        return

    mfccs_60 = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20)
    feature_mfccs_60_stats = get_stats_from_feature(mfccs_60)
    stft = np.abs(librosa.stft(X))
    feature_mel_32_stats = get_stats_from_feature(librosa.feature.melspectrogram(y=X, sr=sample_rate,
                                                                                 n_fft=2048, hop_length=512,
                                                                                 n_mels=32, fmax=8000))
    feature_zcr_stats = get_stats_from_feature(librosa.feature.zero_crossing_rate(y=X))
    feature_rms_stats = get_stats_from_feature(librosa.feature.rms(y=X))
    # 将所有特征连接成一个数组
    features = np.concatenate((feature_mfccs_60_stats,
                               feature_mel_32_stats,
                               feature_zcr_stats,
                               feature_rms_stats
                               ), axis=0)
    # 定义特征列名
    # updated at 20240916
    prefixes = {'mfcc': 20, 'mel32': 32, 'zcr': 1, 'rms': 1}
    column_names = []
    for prefix, num_features in prefixes.items():
        for prefix_stats in ['mean', 'median', 'std', 'p10', 'p90']:
            if num_features > 1:
                column_names.extend([f'{prefix}_{prefix_stats}_{i}' for i in range(1, num_features + 1)])
            else:
                column_names.append(f'{prefix}_{prefix_stats}')

    assert len(column_names) == 5 * (20 + 32 + 2)

    feature_part1 = {}
    for key, value in zip(column_names, features):
        feature_part1[key] = value

    sound = parselmouth.Sound(values=X, sampling_frequency=sample_rate, start_time=0)
    intensity_attributes = get_intensity_attributes(sound)[0]
    pitch_attributes = get_pitch_attributes(sound)[0]
    spectrum_attributes = get_spectrum_attributes(sound)[0]
    expanded_intensity_attributes = {f"Intensity_{key}": value for key, value in intensity_attributes.items()}
    expanded_pitch_attributes = {f"Pitch_{key}": value for key, value in pitch_attributes.items()}
    expanded_spectrum_attributes = {f"Spectrum_{key}": value for key, value in spectrum_attributes.items()}

    feature_prosody = {
        **expanded_intensity_attributes,  # Unpack expanded intensity attributes
        **expanded_pitch_attributes,  # Unpack expanded pitch attributes
        **expanded_spectrum_attributes,  # Unpack expanded spectrum attributes
    }
    feature_combined = {**feature_part1, **feature_prosody}
    # print("feature_combined:",feature_combined)
    return feature_combined


def calc_feature_all_from_binary(x: np.ndarray):
    sample_rate = 16000

    mfccs_20 = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=20)
    feature_mfccs_20_stats = get_stats_from_feature(mfccs_20)
    stft = np.abs(librosa.stft(x))
    feature_mel_32_stats = get_stats_from_feature(librosa.feature.melspectrogram(y=x, sr=sample_rate,
                                                                                 n_fft=2048, hop_length=512,
                                                                                 n_mels=32, fmax=8000))
    feature_zcr_stats = get_stats_from_feature(librosa.feature.zero_crossing_rate(y=x))
    feature_rms_stats = get_stats_from_feature(librosa.feature.rms(y=x))
    # 将所有特征连接成一个数组
    features = np.concatenate((feature_mfccs_20_stats,
                               feature_mel_32_stats,
                               feature_zcr_stats,
                               feature_rms_stats
                               ), axis=0)
    # 定义特征列名
    # updated at 20240916
    prefixes = {'mfcc': 20, 'mel32': 32, 'zcr': 1, 'rms': 1}
    column_names = []
    for prefix, num_features in prefixes.items():
        for prefix_stats in ['mean', 'median', 'std', 'p10', 'p90']:
            if num_features > 1:
                column_names.extend([f'{prefix}_{prefix_stats}_{i}' for i in range(1, num_features + 1)])
            else:
                column_names.append(f'{prefix}_{prefix_stats}')

    assert len(column_names) == 5 * (20 + 32 + 2)

    feature_part1 = {}
    for key, value in zip(column_names, features):
        feature_part1[key] = value

    sound = parselmouth.Sound(values=x, sampling_frequency=sample_rate, start_time=0)
    intensity_attributes = get_intensity_attributes(sound)[0]
    pitch_attributes = get_pitch_attributes(sound)[0]
    spectrum_attributes = get_spectrum_attributes(sound)[0]
    expanded_intensity_attributes = {f"Intensity_{key}": value for key, value in intensity_attributes.items()}
    expanded_pitch_attributes = {f"Pitch_{key}": value for key, value in pitch_attributes.items()}
    expanded_spectrum_attributes = {f"Spectrum_{key}": value for key, value in spectrum_attributes.items()}

    feature_prosody = {
        **expanded_intensity_attributes,  # Unpack expanded intensity attributes
        **expanded_pitch_attributes,  # Unpack expanded pitch attributes
        **expanded_spectrum_attributes,  # Unpack expanded spectrum attributes
    }
    feature_combined = {**feature_part1, **feature_prosody}
    # print("feature_combined:",feature_combined)
    return feature_combined

def preprocess_signal(x_input):
    sample_rate = 16000  # Example sample rate
    min_duration_sec = 0.2  # Minimum duration in seconds
    min_duration_samples = int(min_duration_sec * sample_rate)  # Convert to samples
    max_duration_sec = 5  # Max duration in seconds
    max_duration_samples = int(max_duration_sec * sample_rate)  # Convert to samples

    # 检查音频是否为空
    if len(x_input) == 0:
        print(f"Skipping because input is empty.")
        return

    # 获取音频的实际时长 to be removed - only debug used
    audio_duration = librosa.get_duration(y=x_input, sr=sample_rate)
    print(f"Audio duration for : {audio_duration:.2f} seconds")
    # 丢弃小于 0.128 秒的音频文件
    if audio_duration < 0.128:
        print(f"Skipping because input is too short (<0.128s).")
        return
    if audio_duration > 5:
        print(f"[WARNING] input binary signal last more than 5 seconds.")

    # Determine the number of samples in the current audio
    current_samples = len(x_input)
    # If the audio is shorter than the minimum duration, pad it with zeros
    if current_samples < min_duration_samples:
        padding_samples = min_duration_samples - current_samples
        # Pad with zeros at the end of the audio signal
        x = np.pad(x_input, (0, padding_samples), mode='constant')
        print(f"Audio was padded to {min_duration_sec} seconds")
    elif current_samples > max_duration_samples:
        x = x_input[:max_duration_samples]
    else:
        x = x_input  # No padding needed

    return x

def audio_model_inference(x_input: np.ndarray):
    try:
        x = preprocess_signal(x_input)
        # TODO 5 second windows
        feature_test_instance = calc_feature_all_from_binary(x)
        test_instance = [feature_test_instance[key] for key in selected_feature_name if key in feature_test_instance]
        if not feature_test_instance:
            print("[ATTENTION] - feature_test_instance is none:")
            return None, None
        # last semester score
        final_score = calculate_final_score(test_instance)
        # this semester score - replace [-1,0,1] with scaled max_prob * [-1,0,1]
        sentiment_class_3_new, sentiment_3_new_score = CNN_Model_Predication_New(test_instance)

        # if sentiment_class_3_new is list，then pick the first one
        if isinstance(sentiment_class_3_new, list):
            sentiment_class_3_new = sentiment_class_3_new[0]

        combine_score = calculate_combine_score(test_instance, final_score, sentiment_3_new_score)

        sentiment_category = determine_sentiment_category(sentiment_class_3_new)
        if isinstance(combine_score, (int, float)):  # Check if it's an int or float
            return float(combine_score), sentiment_category
        else:
            return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e} at audio_model_inference()")
        return None, None

"""## Boosting Model Predication"""

model_file_dir='./models'

def Boosting_Model_Predication(test_instance):
    pickle_file_path = f"{model_file_dir}/HistGradientBoostingClassifier_model_8cls_128feat_70acc.pkl"
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"Model file not found at: {pickle_file_path}")

    try:
        with open(pickle_file_path, 'rb') as file:
            gb_fast_classifier = pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from file: {pickle_file_path}. Error: {e}")

    try:
        predication = gb_fast_classifier.predict(np.array(test_instance).reshape(1, -1))
        # print(predication)
        return predication
    except Exception as e:
        raise Exception(f"Error during model prediction: {e}")

def Boosting_Model_Predication_New(test_instance):
    pickle_file_path = f"{model_file_dir}/DecisionTree_rank_2_accuracy55_20240915_160915.pkl"
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"Model file not found at: {pickle_file_path}")

    try:
        with open(pickle_file_path, 'rb') as file:
            gb_fast_classifier = pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from file: {pickle_file_path}. Error: {e}")

    try:
        predication = gb_fast_classifier.predict(np.array(test_instance).reshape(1, -1))
        print(predication)
        return predication
    except Exception as e:
        raise Exception(f"Error during model prediction: {e}")


"""## CNN Model Predication"""


#last semester
def CNN_Model_Predication(test_instance):
    model = EMO_CNN_MODEL #load_model("./models/T2-0329-aug-VAL-R10-6042.h5", compile=False)
    X_test = test_instance
    X_test_cnn = np.expand_dims(X_test, axis=0).astype(np.float32)

    y_pred = np.argmax(model.predict(X_test_cnn), axis=-1)
    print("CNN Model Predication:",y_pred)
    return y_pred

# CNN model prediction  ---this semester
def CNN_Model_Predication_New(test_instance):
    model = NCS_SEN_CNN_MODEL  # load_model("./models/NCS_SEN_CNN_T2_S1S3S2Aa_0916-B-7805.h5", compile=False)
    X_test = test_instance
    X_test_cnn = np.expand_dims(X_test, axis=0).astype(np.float32)

    # Get the predicted probabilities from the softmax layer
    y_pred_probs = model.predict(X_test_cnn)
    # print("Softmax probabilities:", y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    # y_pred_coefficient = np.interp(np.max(y_pred_probs), (0.2, 0.7), (0, 1))
    # y_pred = np.argmax(model.predict(X_test_cnn), axis=-1)

    if y_pred == 0:
        sentiment_class_3_new = -1
    elif y_pred == 1:
        sentiment_class_3_new = 0
    elif y_pred == 2:
        sentiment_class_3_new = 1
    # print("this semester CNN Model output:", sentiment_class_3_new)
    return sentiment_class_3_new, round(y_pred * sentiment_class_3_new, 4)

"""## singlish_model_inference"""
def Singlish_Model_Predication(test_instance):
    model = NCA_LAN_MLP_MODEL # load_model("./models/NCS_LAN_MLP_V2_0916-A2-9722.h5", compile=False)
    X_test = test_instance
    X_test_cnn = np.expand_dims(X_test, axis=0).astype(np.float32)

    y_pred = np.argmax(model.predict(X_test_cnn), axis=-1)
    print("Singlish Model Predication:",y_pred)
    return y_pred


"""## Random Forest Predication
## Return max_prob
"""

import pickle
import numpy as np


def retrieve_max_prob_random_forest(test_instance):
    # Load the Random Forest model
    with open(f"{model_file_dir}/HistGradientBoostingClassifier_model_8cls_128feat_70acc.pkl", 'rb') as file:
        rf_classifier = pickle.load(file)

    # Make predictions on the test instance
    predication1 = rf_classifier.predict(np.array(test_instance).reshape(1, -1))
    predictions_proba = rf_classifier.predict_proba(np.array(test_instance).reshape(1, -1))
    # print(predication1)

    # Get the highest probability for each prediction
    max_prob = predictions_proba.max(axis=1)
    # Round the maximum probability to four decimal places
    max_prob_rounded = np.round(max_prob, 4)

    # return predication1,max_prob
    return max_prob_rounded

# region score mapping and weightages aggregation
"""## mapping 8 class to 3 class"""

# last semester,PKL output is char
def determine_sentiment(predictions):
    for prediction in predictions:
        if prediction in ["Anger", "Disgust", "Fear", "Sadness"]:
            sentiment_final = "Negative"  # Negative sentiment
        elif prediction in ["Calmness", "Neutrality"]:
            sentiment_final = "Neutral"  # Neutral sentiment
        elif prediction in ["Surprise", "Happiness"]:
            sentiment_final = "Positive"  # Positive sentiment
    return sentiment_final


# determine sentiment category based on combine score
def determine_sentiment_category(combine_score):
    sentiment_category = "Neutral sentiment"  # default Neutral
    if combine_score < -0.3:
        sentiment_category = "Negative sentiment"
    elif -0.3 <= combine_score <= 0.3:
        sentiment_category = "Neutral sentiment"
    elif combine_score > 0.3:
        sentiment_category = "Positive sentiment"
    print("determine sentiment category:", sentiment_category)
    return sentiment_category


weighted_score = 0
"""## Calculation final Score"""
def calculate_final_score(test_instance):
    max_prob = retrieve_max_prob_random_forest(test_instance)
    print("retrieve max prob:",max_prob)
    model_predicate=Boosting_Model_Predication(test_instance)
    print("last semester boosting model predicate:", model_predicate)
    # print("max probability is: {}".format(max_prob))
    # calculate confidence_value
    if max_prob > 0.7:
        confidence_value = 1
    elif max_prob < 0.2:
        confidence_value = 0
    else:
        coefficient = np.interp(max_prob, (0.2, 0.7), (0, 1))
        confidence_value = coefficient[0]
    # print("confidence value is: {}".format(confidence_value))
    # calculate weighted_score
    if model_predicate == "Anger":
        weighted_score = -1
    elif model_predicate == "Sadness":
        weighted_score = -0.75
    elif model_predicate == "Disgust" or model_predicate == "Fear":
        weighted_score = -0.5
    elif model_predicate == "Neutrality" or model_predicate == "Calmness":
        weighted_score = 0
    elif model_predicate == "Surprise":
        weighted_score = 0.5
    elif model_predicate == "Happiness":
        weighted_score = 1

    score = weighted_score * confidence_value

    # Round the final score to four decimal places
    final_score = np.round(score, 4)
    print("last semester final score:", final_score)
    return final_score


def calculate_combine_score(test_instance,final_score,sentiment_class_3_new):
    singlish_output = Singlish_Model_Predication(test_instance)

    if singlish_output == 0:  # Singlish output
        weight_s1 = 0.2
        weight_s2 = 0.8
    else:
        weight_s1 = 0.8
        weight_s2 = 0.2
    combine_score = final_score * weight_s1 + sentiment_class_3_new * weight_s2
    print("Final combine score:", combine_score)
    return combine_score

# endregion


"""## Retrieve Probability from RF"""

def retrieve_probability(test_instance):
    # Load the Random Forest model
    with open(f"{model_file_dir}/RandomForestClassifier_model_8cls_131feat_56acc.pkl", 'rb') as file:
        rf_classifier = pickle.load(file)

    # Make predictions on the test instance
    predictions_proba = rf_classifier.predict_proba(np.array(test_instance).reshape(1, -1))
    class_names = rf_classifier.classes_
    probability_dict = [{"Filename": class_name, "Final_Score": prob} for class_name, prob in zip(class_names, predictions_proba[0])]
    return probability_dict


"""
###  MAIN
### Single wav file predication
### need uncomment code below,需要把上面循环的代码注释掉
"""

#
# from tensorflow.keras.models import load_model
# import numpy as np


# Main execution flow
# if __name__ == "__main__":
#     wav_file_path = "./test_files/app_0016_0001_phnd_cc-hot_190.91_195.68.wav"
#     feature_test_instance = calc_feature_all(wav_file_path)
#     test_instance = [feature_test_instance[key] for key in selected_feature_name if key in feature_test_instance]
#
#     final_score= calculate_final_score(test_instance)
#
#     sentiment_class_3_new= CNN_Model_Predication_New(test_instance)
#
#     score=calculate_combine_score(test_instance,final_score,sentiment_class_3_new)


    # test this semester pkl
    # result=Boosting_Model_Predication(test_instance)
    # print(result)

    # test CNN model
    # sentiment_class_3_new = CNN_Model_Predication_New(test_instance)
    # sentiment_category = determine_sentiment_category(sentiment_class_3_new)









