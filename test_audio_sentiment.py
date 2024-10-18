import librosa
from model_predicate import *

wave_file_path = "./speaker/agent_0001.wav"
if __name__ == "__main__":
    feature_test_instance = calc_feature_all(wave_file_path)
    test_instance = [feature_test_instance[key] for key in selected_feature_name if key in feature_test_instance]
    print("length of the vector list: ", len(test_instance))
    sentiment_class_CNN, sentiment_score_CNN = CNN_Model_Predication_New(test_instance)
    print("CNN done; ", end='')
    sentiment_class_RF, sentiment_score_RF = pickle_model_predict(RF_CLS_MODEL, test_instance)
    print("RF done; ", end='')
    sentiment_class_LGBM, sentiment_score_LGBM = pickle_model_predict(LGBM_CLS_MODEL, test_instance)
    print("[SCORE] CNN {:.2f}   RF {:.2f}   LGBM {:.2f}".format(
        sentiment_score_CNN, sentiment_score_RF, sentiment_score_LGBM))
    combine_score = (sentiment_score_CNN + sentiment_score_RF + sentiment_score_LGBM) / 3
    print("[SCORE] final ".format(combine_score))
    sentiment_category = determine_sentiment_category(combine_score)
    # print("[TIME] - takes {:.2f} seconds".format(time.time() - start))
    if isinstance(combine_score, (int, float)):  # Check if it's an int or float
        print(float(combine_score), sentiment_category)
    else:
        print()