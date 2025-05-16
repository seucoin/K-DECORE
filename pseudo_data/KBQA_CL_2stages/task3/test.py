from sql_metadata import Parser
from collections import defaultdict
import re

def format_sql_metadata(sql):
    parser = Parser(sql)
    # 获取 SQL 中涉及的表（顺序由 sql_metadata 返回）
    tables = parser.tables  
    alias_mapping = parser.tables_aliases
    raw_columns = parser.columns

    # 用列表保存各表的列，保持原始顺序（避免使用 set）
    table_columns_map = defaultdict(list)
    identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

    # 根据 sql_metadata 提取的列信息构造映射
    for column in raw_columns:
        # 跳过函数调用或全选
        if '(' in column or ')' in column or column.strip() == '*':
            continue
        parts = column.split('.')
        if len(parts) == 2:
            alias, col = parts
            table = alias_mapping.get(alias, alias)
            if identifier_re.match(col) and col not in table_columns_map[table]:
                table_columns_map[table].append(col)
        else:
            # 当列中没有表名前缀时：如果只有一个表则归属该表，否则对所有表添加
            col = parts[0]
            if len(tables) == 1:
                if identifier_re.match(col) and col not in table_columns_map[tables[0]]:
                    table_columns_map[tables[0]].append(col)
            else:
                for t in tables:
                    if identifier_re.match(col) and col not in table_columns_map[t]:
                        table_columns_map[t].append(col)

    # 构造字面量映射 literal_map[table][column] = list(字面量)
    literal_map = defaultdict(lambda: defaultdict(list))
    # 支持 "=" 或 "!=" 的条件（针对有前缀的列）
    pattern_qualified = r'(\w+)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=)\s*([\'"])(.*?)\3'
    for alias, col, quote, literal in re.findall(pattern_qualified, sql):
        table = alias_mapping.get(alias, alias)
        if literal not in literal_map[table][col]:
            literal_map[table][col].append(literal)
    # 针对无前缀的条件
    pattern_unqualified = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=)\s*([\'"])(.*?)\2'
    # 仅在只有一个表时，将无前缀条件归于该表
    if len(tables) == 1:
        for col, quote, literal in re.findall(pattern_unqualified, sql):
            if literal not in literal_map[tables[0]][col]:
                literal_map[tables[0]][col].append(literal)

    # 根据 literal_map 对表中对应列进行格式更新
    for table in table_columns_map:
        new_cols = []
        for col in table_columns_map[table]:
            if col in literal_map[table] and literal_map[table][col]:
                # 用括号注释出条件中的字面量，多个字面量以逗号分隔
                new_cols.append(f"{col} ( {', '.join(literal_map[table][col])} )")
            else:
                new_cols.append(col)
        table_columns_map[table] = new_cols

    # 构造最终输出，保持 sql_metadata 返回的表顺序
    output_parts = []
    for table in tables:
        if table in table_columns_map and table_columns_map[table]:
            columns_str = ", ".join(table_columns_map[table])
            output_parts.append(f"{table} : {columns_str}")
        else:
            output_parts.append(table)
    formatted_output = " | ".join(output_parts)
    return formatted_output

# 测试示例
sql1 = "select name from head where born_state != 'California'"
sql2 = 'select t1.name from employee as t1 join certificate as t2 on t1.eid = t2.eid join aircraft as t3 on t3.aid = t2.aid where t3.name = "Boeing 737-800" intersect select t1.name from employee as t1 join certificate as t2 on t1.eid = t2.eid join aircraft as t3 on t3.aid = t2.aid where t3.name = "Airbus a340-300"'
sql3 = 'select t1.name from employee as t1 join certificate as t2 on t1.eid = t2.eid join aircraft as t3 on t3.aid = t2.aid where t3.name = "Boeing 737-800"'
sql4 = "select t1.name, t1.num_employees from department as t1 join management as t2 on t1.department_id = t2.department_id where t2.temporary_acting = 'Yes'"
sql5 = "select count(*) from farm"

print(format_sql_metadata(sql1))
# 正确输出应为：
# "head : name, born_state ( California )"

print(format_sql_metadata(sql2))
# 预期输出：
# "aircraft : aid, name ( Airbus a340-300, Boeing 737-800 ) | employee : eid, name | certificate : eid"

print(format_sql_metadata(sql3))
# 预期输出：
# "aircraft : aid, name ( Boeing 737-800 ) | employee : eid, name | certificate : eid"

print(format_sql_metadata(sql4))
# 预期输出：
# "department : department_id, name, num_employees | management : department_id, temporary_acting ( Yes )"

print(format_sql_metadata(sql5))
# 预期输出："farm"
