"""ReadOutputDemo.py"""

import json

file = open('YOUR_INPUT_FILE_PATH', 'r')


data = {}

# read in data (two parameters each line) and print
for line in file:

	lineline = line.rstrip('\n')
	pair = lineline.split(" ")
	id = pair[0]
	path = pair[1]
	print "id: " + id
	print "path: " + path

	data[id] = path

# write data into a json file
json.dump(data, open('YOUR_OUTPUT_FILE_PATH','w'))
