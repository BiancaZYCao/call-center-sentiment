# -*- coding: utf-8 -*-

import os
import warnings

import h5py

print(h5py.__version__)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
VERSION = 4
RANDOM_SEED = 7

from tensorflow.keras.models import load_model
from model_predicate import *
"""## Load data - To Define your selected features
## change to be aligned with your model input!
"""


df_joint_train_aug  = pd.read_csv('feature_name_load.csv', low_memory=False)
feature_column_names = [i for i in df_joint_train_aug.columns \
                        if i not in ['file_path','renamed_file_path','split','sentiment_value','emotional_category']]


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

# print(selected_feature_name)



def test_singlish_detection(directory="./test_lang"):
    # Loop through all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith("_5.wav"):  # Check if the file is a .wav file
            wave_file_path = os.path.join(directory, file_name)  # Full file path

            # Calculate features for the current file
            try:
                feature_test_instance = calc_feature_all(wave_file_path)
                test_instance = [feature_test_instance[key] for key in selected_feature_name if
                                 key in feature_test_instance]

                # Make a prediction with the Singlish model
                singlish_output = Singlish_Model_Predication(test_instance)

                # Check if the output is Singlish (assuming 0 means Singlish, adjust if needed)
                if singlish_output == 0:
                    print(f"Result for {file_name}: Singlish")
                else:
                    print(f"Result for {file_name}: Not Singlish")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")




if __name__ == "__main__":
    # Call the test function
    test_singlish_detection()