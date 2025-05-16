import os
import json

from SPARQLWrapper import SPARQLWrapper, JSON

def execute_sparql_query(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results

def parse_query_results(response):

    if "boolean" in response:
        results = [response["boolean"]]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]: # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                res = [v["value"].replace('http://rdf.freebase.com/ns/', "") for k, v in res.items()]
                results.extend(res)
    return results

def query_ent_name(x, kb_endpoint):
    query = (
        "PREFIX ns: <http://rdf.freebase.com/ns/> "
        "SELECT ?name WHERE { " + x + " ns:type.object.name ?name . "
        "FILTER (lang(?name) = 'en') }"
    )
    results = execute_sparql_query(query, kb_endpoint)
    if len(results) == 0:
        query = (
            "PREFIX ns: <http://rdf.freebase.com/ns/> "
            "SELECT ?name WHERE { " + x + " ns:common.topic.alias ?name . "
            "FILTER (lang(?name) = 'en') }"
        )
        results = execute_sparql_query(query, kb_endpoint)
        if len(results) == 0:
            print(x, "does not have name !")
            return x
        
    name = results[0]
    return name

relations = json.load(open('./grailqa.relations.json'))
types = json.load(open('./grailqa.types.json'))

relation_set = set(relations)
type_set = set(types)


with open('./origin_compwebq_test.json', 'r') as f:
    data = json.load(f)

with open('./grailqa.entity_dict.test.json', 'r') as f:
    entity_dict = json.load(f)

with open('./ComplexWebQuestions_dev.json', 'r') as f:
    dev_data = json.load(f)

dev_mapping = {entry["question"]: entry["sparql"] for entry in dev_data}

entity_output_list = []
full_schema_list = []

def process_data(entry):
    schema = entry['instruction'].split('knowledge graph triples:\n\n')[1].split('\n\n\nquestion:\n\n')[0]
    triples = schema.split(' | ')
    table_column_mapping = {}

    for triple in triples:
        items = triple.split()
        for item in items:
            if ".." in item:
                item1, item2 = item.split("..")
                if item1 in relation_set:
                    t_name, c_name = item1.rsplit('.', 1)
                    if t_name not in table_column_mapping:
                        table_column_mapping[t_name] = []
                    table_column_mapping[t_name].append(c_name)       
                if item2 in relation_set:
                    t_name, c_name = item2.rsplit('.', 1)
                    if t_name not in table_column_mapping:
                        table_column_mapping[t_name] = []
                    table_column_mapping[t_name].append(c_name)       
            else:
                if item in relation_set:
                    t_name, c_name = item.rsplit('.', 1)
                    if t_name not in table_column_mapping:
                        table_column_mapping[t_name] = []
                    table_column_mapping[t_name].append(c_name)   

        for table, columns in table_column_mapping.items():
            columns = list(set(columns))
            table_column_mapping[table] = columns

    schema_output = ' | '.join([f"{table} : {', '.join(columns)}" for table, columns in table_column_mapping.items()])
    question = entry['instruction'].split('question:\n\n')[1]
    full_schema_list.append(schema_output)
    
    instruction = f"Identify the exact schema component from the given schema that would correctly answer the following question. schema:\n\n{schema_output}\n\n\nquestion:\n\n{question}"
    entry['instruction'] = instruction

    sparql = dev_mapping.get(question)
    entry['output'] = sparql    

    output = entry["output"]
    output = output.replace("ns: <http://rdf.freebase.com/ns/>", "").replace("(", " ").replace(")", " ")
    keywords = []
    for word in output.split():
        if word.startswith('ns:'):
            keyword = word[3:]
            keywords.append(keyword)
    
    formatted_keywords = []

    table_column_mapping = {}

    for keyword in keywords:
        if keyword.startswith('m.'):
            find_flag = False
            for entity_group in entity_dict.values():
                if keyword in entity_group.values():
                    original_entity = next(name for name, ref in entity_group.items() if ref == keyword)
                    formatted_keywords.append(f"{original_entity}: {keyword}")
                    find_flag = True
                    break
            if not find_flag:
                original_entity = query_ent_name('ns:'+keyword, "http://10.201.234.130:3001/sparql")
                formatted_keywords.append(f"{original_entity}: {keyword}")

        else: 
            if keyword in relation_set:
                table_name, column_name = keyword.rsplit('.', 1)
                if table_name not in table_column_mapping:
                    table_column_mapping[table_name] = []
                table_column_mapping[table_name].append(column_name)   

            else:
                table_name = keyword
                if table_name not in table_column_mapping:
                    table_column_mapping[table_name] = []

            for table, columns in table_column_mapping.items():
                columns = list(set(columns))
                table_column_mapping[table] = columns         


    unique_keywords = []
    seen = set()
    for item in formatted_keywords:
        if item not in seen:
            seen.add(item)
            unique_keywords.append(item)

    entity_output = ' | '.join(unique_keywords)
    entity_output_list.append(entity_output)

    schema_output = ' | '.join([f"{table} : {', '.join(columns)}" if columns else f"{table}" for table, columns in table_column_mapping.items()])
    new_output = schema_output
    entry["output"] = new_output

    return entry

processed_data = [process_data(entry) for entry in data]

with open('compwebq_test1.json', 'w') as f:
    json.dump(processed_data, f, indent=4)

with open('ComplexWebQuestions_dev.json', 'r', encoding='utf-8') as complex_file:
    complex_data = json.load(complex_file)

with open('origin_compwebq_test.json', 'r') as f:
    test_data = json.load(f)

model_paths = {
    "M4": {
        "schema_file": "../../../saves/KBQA_CL_2stages_order1_qwen2.5/qwen2.5/seq_lora/stage1/task4/result/test4_stage1.json",
        "output_file": "compwebq_test2_M4.json"
    }
}


def process_test_data(test_data, new_schema_list):
    guidance = (
        "Please prioritize using the candidate schema to generate an SPARQL query. "
        "If the candidate schema is insufficient, you may refer to the full schema for additional context. "
    )
    processed_data = []
    for i, entry in enumerate(test_data):
        instruction = entry['instruction']
        part1 = "Given the list of schema items, write an SPARQL query that can be used to find the answer to the following question. " + guidance
        part2 = '\n\n\n' + instruction.split('\n\n\n')[1]
        if entity_output_list[i] == '':
            entry['instruction'] = part1 + "full schema:\n\n" + full_schema_list[i] + "\n\n" + "candidate schema:\n\n" + new_schema_list[i] + part2
        else:
            entry['instruction'] = part1 + "full schema:\n\n" + full_schema_list[i] + "\n\n" + "candidate schema:\n\n" + entity_output_list[i] + " | " + new_schema_list[i] + part2
        
        for complex_item in complex_data:
            if complex_item.get("ID") == entry.get("id"):
                entry["output"] = complex_item.get("sparql")
                break
        processed_data.append(entry)
    return processed_data

for model, paths in model_paths.items():
    with open(paths["schema_file"], 'r') as f:
        new_schema_list = json.load(f)
    processed_data = process_test_data(test_data, new_schema_list)

    with open(paths["output_file"], 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"{model} finished, save into {paths['output_file']}")