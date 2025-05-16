import os
import json
import re
from collections import defaultdict
from sql_metadata import Parser


full_schema_list = []

with open('./spider_with_cell_test.json', 'r') as f:
    data = json.load(f)


def format_sql_metadata(sql):
    parser = Parser(sql)
    tables = parser.tables  
    alias_mapping = parser.tables_aliases
    raw_columns = parser.columns

    table_columns_map = defaultdict(list)
    identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
    for column in raw_columns:
        if '(' in column or ')' in column or column.strip() == '*':
            continue
        parts = column.split('.')
        if len(parts) == 2:
            alias, col = parts
            table = alias_mapping.get(alias, alias)
            if identifier_re.match(col) and col not in table_columns_map[table]:
                table_columns_map[table].append(col)
        else:
            col = parts[0]
            if len(tables) == 1:
                if identifier_re.match(col) and col not in table_columns_map[tables[0]]:
                    table_columns_map[tables[0]].append(col)
            else:
                for t in tables:
                    if identifier_re.match(col) and col not in table_columns_map[t]:
                        table_columns_map[t].append(col)

    literal_map = defaultdict(lambda: defaultdict(list))
    pattern_qualified = r'(\w+)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=)\s*([\'"])(.*?)\3'
    for alias, col, quote, literal in re.findall(pattern_qualified, sql):
        table = alias_mapping.get(alias, alias)
        if literal not in literal_map[table][col]:
            literal_map[table][col].append(literal)

    pattern_unqualified = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=)\s*([\'"])(.*?)\2'

    if len(tables) == 1:
        for col, quote, literal in re.findall(pattern_unqualified, sql):
            if literal not in literal_map[tables[0]][col]:
                literal_map[tables[0]][col].append(literal)

    for table in table_columns_map:
        new_cols = []
        for col in table_columns_map[table]:
            if col in literal_map[table] and literal_map[table][col]:
                new_cols.append(f"{col} ( {', '.join(literal_map[table][col])} )")
            else:
                new_cols.append(col)
        table_columns_map[table] = new_cols

    output_parts = []
    for table in tables:
        if table in table_columns_map and table_columns_map[table]:
            columns_str = ", ".join(table_columns_map[table])
            output_parts.append(f"{table} : {columns_str}")
        else:
            output_parts.append(table)
    formatted_output = " | ".join(output_parts)
    return formatted_output

def process_data(entry):
    schema_section = entry['instruction'].split('database schema:\n\n')[1].split('\n\n\n')[0]
    schema_section = " " + schema_section
    schema_section = " | ".join(schema_section.split(' | ')[2:])

    question = entry['instruction'].split('question:\n\n')[1]
    
    instruction = f"Identify the exact schema component from the given schema that would correctly answer the following question. schema:\n\n{schema_section}\n\n\nquestion:\n\n{question}"
    entry['instruction'] = instruction
    full_schema_list.append(schema_section)

    output = entry['seq_out']
    new_output = format_sql_metadata(output)
    entry['output'] = new_output

    return entry


processed_data = [process_data(entry) for entry in data]

with open('spider_test1.json', 'w') as f:
    json.dump(processed_data, f, indent=4)



model_paths = {
    "M3": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task3/result/test3_stage1.json",
        "output_file": "spider_test2_M3.json"
    },
    "M4": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task4/result/test3_stage1.json",
        "output_file": "spider_test2_M4.json"
    }
}

def process_test_data(test_data, new_schema_list):
    guidance = (
        "Please prioritize using the candidate schema to generate an SQL query. "
        "If the candidate schema is insufficient, you may refer to the full schema for additional context. "
    )
    processed_data = []
    for i, entry in enumerate(test_data):
        instruction = entry['instruction']
        part1 = instruction.split('database schema:\n\n')[0] + guidance
        part2 = '\n\n\n' + instruction.split('\n\n\n')[1]
        entry['instruction'] = part1  + "full schema:\n\n" + full_schema_list[i] + "\n\n" + "candidate schema:\n\n" + new_schema_list[i] + part2
        processed_data.append(entry)
    return processed_data

for model, paths in model_paths.items():
    with open('spider_with_cell_test.json', 'r') as f:
        test_data = json.load(f)

    with open(paths["schema_file"], 'r') as f:
        new_schema_list = json.load(f)

    processed_data = process_test_data(test_data, new_schema_list)

    with open(paths["output_file"], 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"{model} finished, save into {paths['output_file']}")