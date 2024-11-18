""" score adjustment functions """


def adjust_audio_scores(audio_score_list_slice, text_sentiment_result):
    """ adjust audio scores (time series) by text sentiment results """
    average_audio_score = round(sum(audio_score_list_slice) / len(audio_score_list_slice), 2)

    if text_sentiment_result == 'Negative' and average_audio_score > 0:
        audio_score_list_slice = [-abs(score) for score in audio_score_list_slice]
    elif text_sentiment_result == 'Positive' and average_audio_score < 0:
        audio_score_list_slice = [abs(score) for score in audio_score_list_slice]
    elif text_sentiment_result == 'Neutral' and abs(average_audio_score) > 0.2:
        audio_score_list_slice = [score * 0.5 for score in audio_score_list_slice]
    return audio_score_list_slice


def update_final_scores(final_score_list_current, end_time_list_current, time_points, new_scores):
    """
    Updates the final_score_list based on the corresponding time_points from end_time_list
    """
    if len(time_points) != len(new_scores):
        # raise ValueError("time_points and new_scores must have the same length")
        return final_score_list_current

    for i, time_point in enumerate(time_points):
        # Find the index in end_time_list that matches the current time_point
        try:
            index = end_time_list_current.index(time_point)  # This finds the index where the time matches
            final_score_list_current[index] = round(new_scores[i], 2)
        except ValueError:
            pass
    return final_score_list_current
