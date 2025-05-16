import os
from collections import defaultdict
import re
import json

INTENTS = [
    'IN:GET_MESSAGE', 'IN:GET_WEATHER', 'IN:GET_ALARM', 'IN:SEND_MESSAGE', 'IN:GET_INFO_RECIPES', 'IN:SET_UNAVAILABLE', 
    'IN:DELETE_REMINDER', 'IN:GET_STORIES_NEWS', 'IN:CREATE_ALARM', 'IN:GET_REMINDER', 'IN:CREATE_REMINDER', 'IN:GET_RECIPES', 
    'IN:QUESTION_NEWS', 'IN:GET_EVENT', 'IN:PLAY_MUSIC', 'IN:GET_CALL_TIME', 'IN:CREATE_CALL', 'IN:END_CALL', 'IN:CREATE_PLAYLIST_MUSIC', 
    'IN:CREATE_TIMER', 'IN:IGNORE_CALL', 'IN:GET_LIFE_EVENT', 'IN:GET_INFO_CONTACT', 'IN:UPDATE_CALL', 'IN:UPDATE_REMINDER_DATE_TIME', 
    'IN:GET_CONTACT', 'IN:GET_TIMER', 'IN:GET_REMINDER_DATE_TIME', 'IN:DELETE_ALARM', 'IN:PAUSE_MUSIC', 'IN:GET_AGE', 'IN:GET_SUNRISE', 
    'IN:GET_EMPLOYER', 'IN:GET_EDUCATION_TIME', 'IN:ANSWER_CALL', 'IN:SET_RSVP_YES', 'IN:SNOOZE_ALARM', 'IN:GET_JOB', 'IN:UPDATE_REMINDER_TODO', 
    'IN:IS_TRUE_RECIPES', 'IN:REMOVE_FROM_PLAYLIST_MUSIC', 'IN:GET_AVAILABILITY', 'IN:GET_CATEGORY_EVENT', 'IN:PLAY_MEDIA', 'IN:ADD_TIME_TIMER', 
    'IN:GET_CALL', 'IN:SET_AVAILABLE', 'IN:ADD_TO_PLAYLIST_MUSIC', 'IN:GET_EMPLOYMENT_TIME', 'IN:SHARE_EVENT', 'IN:PREFER', 'IN:START_SHUFFLE_MUSIC', 
    'IN:GET_CALL_CONTACT', 'IN:GET_LOCATION', 'IN:SILENCE_ALARM', 'IN:SWITCH_CALL', 'IN:GET_TRACK_INFO_MUSIC', 'IN:SUBTRACT_TIME_TIMER', 
    'IN:GET_SUNSET', 'IN:DELETE_TIMER', 'IN:UPDATE_TIMER', 'IN:PREVIOUS_TRACK_MUSIC', 'IN:SET_DEFAULT_PROVIDER_MUSIC', 'IN:HOLD_CALL', 
    'IN:GET_MUTUAL_FRIENDS', 'IN:SKIP_TRACK_MUSIC', 'IN:UPDATE_METHOD_CALL', 'IN:SET_RSVP_INTERESTED', 'IN:QUESTION_MUSIC', 'IN:GET_UNDERGRAD', 
    'IN:PAUSE_TIMER', 'IN:UPDATE_ALARM', 'IN:GET_REMINDER_LOCATION', 'IN:GET_ATTENDEE_EVENT', 'IN:LIKE_MUSIC', 'IN:RESTART_TIMER', 
    'IN:RESUME_TIMER', 'IN:MERGE_CALL', 'IN:GET_MESSAGE_CONTACT', 'IN:REPLAY_MUSIC', 'IN:LOOP_MUSIC', 'IN:GET_REMINDER_AMOUNT', 
    'IN:GET_DATE_TIME_EVENT', 'IN:STOP_MUSIC', 'IN:GET_DETAILS_NEWS', 'IN:GET_EDUCATION_DEGREE', 'IN:SET_DEFAULT_PROVIDER_CALLING', 
    'IN:GET_MAJOR', 'IN:UNLOOP_MUSIC', 'IN:GET_CONTACT_METHOD', 'IN:SET_RSVP_NO', 'IN:UPDATE_REMINDER_LOCATION', 'IN:RESUME_CALL', 
    'IN:CANCEL_MESSAGE', 'IN:RESUME_MUSIC', 'IN:UPDATE_REMINDER', 'IN:DELETE_PLAYLIST_MUSIC', 'IN:REWIND_MUSIC', 'IN:REPEAT_ALL_MUSIC', 
    'IN:FAST_FORWARD_MUSIC', 'IN:DISLIKE_MUSIC', 'IN:GET_LIFE_EVENT_TIME', 'IN:DISPREFER', 'IN:REPEAT_ALL_OFF_MUSIC', 'IN:HELP_REMINDER', 
    'IN:GET_LYRICS_MUSIC', 'IN:STOP_SHUFFLE_MUSIC', 'IN:GET_AIRQUALITY', 'IN:GET_LANGUAGE', 'IN:FOLLOW_MUSIC', 'IN:GET_GENDER', 
    'IN:CANCEL_CALL', 'IN:GET_GROUP'
]

INTENTS_KEY = ["ADD", "CANCEL", "CREATE", "DELETE", "GET", "PAUSE", "PLAY", "QUESTION", "REPEAT", "RESUME", "SET", "STOP", "UPDATE_REMINDER"] 


def process_intents(intents):
    grouped_intents = defaultdict(list)
    for intent in intents:
        prefix, value = intent.split(":")
        if len(value.split("_")) == 1 or value.split("_")[0] not in INTENTS_KEY:
            table_name = intent
            grouped_intents[table_name].append("")
        else:
            table_name = prefix + ":" + value.split("_")[0]
            column_name = "_".join(value.split("_")[1:])
            grouped_intents[table_name].append(column_name)
    
    result = []
    for table_name, columns in grouped_intents.items():
        columns = [col for col in columns if col]
        if columns:
            result.append(f"{table_name} : " + ", ".join(columns))
        else:
            result.append(table_name)
    
    return " | ".join(result)

processed_intents = process_intents(INTENTS)
new_schema = processed_intents

with open('./mtop_test.json', 'r') as f:
    data = json.load(f)

def extract_keywords(output):
    in_keywords = re.findall(r'IN:[A-Z_]+', output)
    return in_keywords

def process_data(entry):
    question = entry['instruction'].split('natural language query:\n\n')[1]
    instruction = f"Identify the exact schema component from the given schema that would correctly answer the following question. schema:\n\n{new_schema}\n\n\nquestion:\n\n{question}"
    
    entry['instruction'] = instruction
    
    in_keywords = extract_keywords(entry['seq_out'])
    processed_in = process_intents(in_keywords)
    entry['output'] = processed_in if processed_in else ""
    
    return entry


processed_data = [process_data(entry) for entry in data]
with open('mtop_test1.json', 'w') as f:
    json.dump(processed_data, f, indent=4)


model_paths = {
    "M2": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task2/result/test2_stage1.json",
        "output_file": "mtop_test2_M2.json"
    },
    "M3": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task3/result/test2_stage1.json",
        "output_file": "mtop_test2_M3.json"
    },
    "M4": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task4/result/test2_stage1.json",
        "output_file": "mtop_test2_M4.json"
    }
}



def process_test_data(test_data, new_schema_list):
    guidance = (
        "Please prioritize using the candidate API specification to generate an API call in Task Oriented Parsing (TOP) representation. "
        "If the candidate API specification is insufficient, you may refer to the full API specification for additional context. "
    )
    processed_data = []
    for i, entry in enumerate(test_data):
        instruction = entry['instruction']
        part1 = instruction.split('API specification:\n\n')[0] + guidance
        part2 = '\n\n\n' + instruction.split('\n\n\n')[1]
        entry['instruction'] = part1 + "full API specification:\n\n" + new_schema + "\n\n" + "candidate API specification:\n\n" + new_schema_list[i] + part2
        processed_data.append(entry)
    return processed_data

for model, paths in model_paths.items():
    with open('mtop_test.json', 'r') as f:
        test_data = json.load(f)
    with open(paths["schema_file"], 'r') as f:
        new_schema_list = json.load(f)
    processed_data = process_test_data(test_data, new_schema_list)
    with open(paths["output_file"], 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"{model} finished, save into {paths['output_file']}")
