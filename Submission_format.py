import csv
import json



def read_csv(file_name):
    f = open(file_name, 'r')
    next(f)
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append(row.split(','))
    return final_list



results = read_csv('./submission.csv')
json_file = open('./submission.json', "w")


final_result_dict = {}
OBJ_array = []

print(len(results))

OBJ_array_save = []
name_array_save = []

for i, res in enumerate(results):
    if len(res) == 6:
        OBJ_array.append((int(res[1]), int(res[2]), int(res[3]), int(res[4]), float(res[5])))
        if i != len(results):
            if results[i][0] == results[i+1][0]:
                continue
            #name = (res[0].split('_'))[1]
            name = res[0]

            name_array_save.append(name)
            OBJ_array_save.append(tuple(OBJ_array))
            OBJ_array = []



json.dump(dict(zip(tuple(name_array_save), tuple(OBJ_array_save))), json_file)
json_file.close()