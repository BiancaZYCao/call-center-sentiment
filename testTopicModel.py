import pandas as pd
from TopicModel import TopicModel

sentence = "I want to know the monthly credit limits of OCBC credit cards."
sentence = "How do I apply for UOB credit card?"
sentence = "What is the interest rate for UOB credit card?"
#sentence = "sorry does does your bank give out home loan"
#sentence = "So the annual fee is currently waived for the first two years. what about payment wise, payment will be billed monthly right? "
#sentence += "ok I supposed if uh if there's any late payment for example there will be late payment charges and finance charges also?"

sentence = """
    So the annual fee is currently waived for the first two years. what about payment wise, payment will be billed monthly right? 
    ok I supposed if uh if there's any late payment for example there will be late payment charges and finance charges also?
    """

tm = TopicModel()

topics = []
questions = {}
topic_question_responses = {}


# Use getTopics function to get all topics from the input sentence.
# There are 10 topics in the list.
def testGetTopics():
    topics = tm.getTopics(sentence)
    print(topics)

# Use getTopicsAndQuestions function to get all topics and set of questions
# for each topic.
def testGetTopicsAndQuestions():
    taq = tm.getTopicsAndQuestions()
    print(taq)

# Run this function to simulate reading from the imda excel file where toics are 
# identified for every input sentences.
# def testWithTranscript():
#     # Load the Excel file
#     file_path = 'input/imda_conversation.xlsx'
#     df = pd.read_excel(file_path)
#
#     # Extract relevant columns where dialog_type is "bank"
#     df_filtered = df[df['dialog_type'] == 'bank'][['speaker_type', 'text']]
#
#     # Iterate through the rows and implement the logic
#     i = 0
#     while i < len(df_filtered):
#         if df_filtered.iloc[i]['speaker_type'] == 'client':
#             sentence = df_filtered.iloc[i]['text']
#             print("Client:", sentence)
#             topics = tm.getTopics(sentence)
#             if len(topics) > 0:
#                 print(topics)
#             # Check if the next row is an agent
#             if i + 1 < len(df_filtered) and df_filtered.iloc[i + 1]['speaker_type'] == 'agent':
#                 input("Press enter to continue...")
#                 # Skip to the next client after agent
#                 while i + 1 < len(df_filtered) and df_filtered.iloc[i + 1]['speaker_type'] != 'client':
#                     i += 1
#         i += 1


def display_numbered_list(items):
    print("items = ", items)
    print("Select an item by entering the corresponding number:")
    for i, item in enumerate(items, start=1):
        print(f"{i}. {item}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            print("choice = ", choice)
            if 1 <= choice <= len(items):
                selected_item = items[choice - 1]
                print(f"You selected: {selected_item}")

                return selected_item
            else:
                print(f"Please enter a number between 1 and {len(items)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


# Run this function to simulate retrieval of questions and response by selecting the topic
# identified by an input sentence.
def user_demo():
    topics = tm.getTopics(sentence)
    questions = tm.getTopicsAndQuestions()
    sel_topic = display_numbered_list(topics)

    if sel_topic in topics:
        sel_question = display_numbered_list(questions[sel_topic])
        res = tm.getResponseForQuestions(sel_question)
        print(res)
    #print(sel_topic)


####### Function Calls for Testing ########

#testGetTopics()
#testGetTopicsAndQuestions()
# testWithTranscript()
#user_demo()