import os
import json
#from multiprocessing import Process,Queue
#import multiprocessing as mp


def get_question_data(q_id,q_part_id,t_id):
	
	test_file_name="tests.json"
	assert os.path.isfile(test_file_name), "Test file was not found! - "+test_file_name

	with open(test_file_name,'r') as f:
		tests = json.loads(f.read())

	return tests[0]["Q"+str(q_id)]["P"+str(q_part_id)]["T"+str(t_id)]['input'],tests[0]["Q"+str(q_id)]["P"+str(q_part_id)]["T"+str(t_id)]['output']
def answers_equivalent(a1,a2):
	assert not(a1 is None), "Output can not be of type None"
	assert not(a2 is None), "Test case can not be of type None"
	assert isinstance(a1,str), "Output should be of type string"
	assert isinstance(a2,str), "Test case should be of type string"
	if len(a1)!= len(a2):
		return False
	for c in range(len(a2)):
		if (a2[c] != a1[c]):
			return False
	return True

def evaluate_solution(question_id=None,question_part_id=None,function=None,test_case_list=None,verbose=False):
	assert question_id in [1,2,3,4,5,6], "Invalid question id: "+str(question_id)
	assert question_part_id in [1], "Invalid question part id:" +str(question_part_id)
	for test_id in test_case_list:
		assert test_id in [1,2,3,4,5,6,7,8,9,10], "Invalid test id: "+str(test_id)
	total_correct = 0
	total_count = 0
	for test_id in test_case_list:
		total_count=total_count + 1
		input_value, output_value = get_question_data(question_id,question_part_id,test_id)
		if verbose == True:
			print('Running question '+str(question_id)+', part '+str(question_part_id)+', test id: '+str(test_id))
			print('Input:\n'+str(input_value))

		output_obtained = function(input_value)
		if verbose == True:
			print('Obtained output:\n'+str(output_obtained))
		if answers_equivalent(str(output_obtained),str(output_value)):
			if verbose == True:
				print('<<CORRECT>>')
			total_correct = total_correct+1
		else:
			if verbose == True:
				print('<<INCORRECT>> -> Expected output: ')
				print(output_value)
	print('\n----------------------------------')
	print('Total correct outputs:\n'+str(total_correct)+ ' out of '+str(total_count))
