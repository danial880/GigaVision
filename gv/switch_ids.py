import json
import argparse

def modify_json_file(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)

    for item in data:
        if item['category_id'] == 2:
            item['category_id'] = 1
            item['category_name'] = 'vehicle'
        else:
            item['category_id'] = 2
            item['category_name'] = 'person'

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_file', type=str, help='Path to annotations JSON file')
    parser.add_argument('-out','--output_file', type=str, help='Path to results JSON file')
    args = parser.parse_args()
    modify_json_file(args.input_file, args.output_file)
