import os
import json

relations = json.load(open('./grailqa.relations.json'))
types = json.load(open('./grailqa.types.json'))
relation_set = set(relations)
type_set = set(types)


with open('./grailqa_test.json', 'r') as f:
    data = json.load(f)

entity_name_list = []
full_schema_list = []

def process_data(entry):
    schema = entry['instruction'].split('graph triples:\n\n')[1].split('\n\n\n')[0]
    necessary_content = schema.split(' | ')[0]
    entity_name_list.append(necessary_content)
    schema_items = schema.split(' | ')[1:][0].split()

    table_column_mapping = {}

    for item in schema_items:
        table = None
        columns_list = []
        
        if item in relation_set:
            table_name, column_name = item.rsplit('.', 1)
            if table_name not in table_column_mapping:
                table_column_mapping[table_name] = []
            table_column_mapping[table_name].append(column_name)
        elif item in type_set:
            table_name = item
            if table_name not in table_column_mapping:
                table_column_mapping[table_name] = []

        for table, columns in table_column_mapping.items():
            columns = list(set(columns)) 
            table_column_mapping[table] = columns

    schema_output = ' | '.join([f"{table} : {', '.join(columns)}" if columns else f"{table}" for table, columns in table_column_mapping.items()])
    full_schema_list.append(schema_output)
    question = entry['instruction'].split('question:\n\n')[1]
    

    instruction = f"Identify the exact schema component from the given schema that would correctly answer the following question. schema:\n\n{schema_output}\n\n\nquestion:\n\n{question}"

    output = entry['s_expression']
    relevant_tables = set()
    relevant_columns = set()

    output = output.replace('(', '').replace(')', '')
    output_parts = output.split()


    for part in output_parts:
        if part in relation_set:
            table_name, column_name = part.rsplit('.', 1)
            relevant_tables.add(table_name)
            relevant_columns.add(column_name)
        elif part in type_set:
            relevant_tables.add(part)

    relevant_schema = {}
    for table in relevant_tables:
        relevant_schema[table] = []

    for part in output_parts:
        if part in relation_set:
            table_name, column_name = part.rsplit('.', 1)
            if table_name in relevant_schema:
                relevant_schema[table_name].append(column_name)

    relevant_output = []

    for table, columns in relevant_schema.items():
        if columns:
            relevant_output.append(f"{table} : {', '.join(list(set(columns)))}")
        else:
            relevant_output.append(f"{table}")

    relevant_output_str = ' | '.join(relevant_output)

    entry['instruction'] = instruction
    if necessary_content != "":
        entry['output'] = relevant_output_str
    else:
        entry['output'] = relevant_output_str
    return entry

processed_data = [process_data(entry) for entry in data]

with open('grailqa_test1.json', 'w') as f:
    json.dump(processed_data, f, indent=4)




# Generating grailqa_test2.json


model_paths = {
    "M1": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_llama3/llama3/seq_lora/stage1/task1/result/test1_stage1.json",
        "output_file": "grailqa_test2_M1.json"
    },
    "M2": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_llama3/llama3/seq_lora/stage1/task2/result/test1_stage1.json",
        "output_file": "grailqa_test2_M2.json"
    },
    "M3": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_llama3/llama3/seq_lora/stage1/task3/result/test1_stage1.json",
        "output_file": "grailqa_test2_M3.json"
    },
    "M4": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_llama3/llama3/seq_lora/stage1/task4/result/test1_stage1.json",
        "output_file": "grailqa_test2_M4.json"
    }
}

def process_test_data(test_data, new_schema_list):
    guidance = (
        "Please prioritize using the candidate schema to generate the s-expression. "
        "If the candidate schema is insufficient, you may refer to the full schema for additional context. "
    )
    processed_data = []
    for i, entry in enumerate(test_data):
        instruction = entry['instruction']
        part1 = instruction.split(' knowledge graph triples:\n\n')[0] + ". " + guidance
        part2 = '\n\n\n' + instruction.split('\n\n\n')[1]
        if entity_name_list[i] == " ":
            entry['instruction'] = part1 + "full schema:\n\n" + full_schema_list[i] + "\n\n" + "candidate schema:\n\n" + " | " + new_schema_list[i] + part2
        else:
            entry['instruction'] = part1 + "full schema:\n\n" + full_schema_list[i] + "\n\n" + "candidate schema:\n\n" + entity_name_list[i] + " | " + new_schema_list[i] + part2
        processed_data.append(entry)
    return processed_data


for model, paths in model_paths.items():
    with open('grailqa_test.json', 'r') as f:
        test_data_copy = json.load(f)
    with open(paths["schema_file"], 'r') as f:
        new_schema_list = json.load(f)
    processed_data = process_test_data(test_data_copy, new_schema_list)
    with open(paths["output_file"], 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"{model} finished, save into {paths['output_file']}")