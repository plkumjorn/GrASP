#!/usr/bin/env python3

GRASP_JSON_PATH = '../patterns_example.json'

from flask import Flask, render_template, request
import json
import math

app = Flask(__name__)

@app.route("/")
@app.route("/summary")
def show_summary():
	return render_template('patterns_summary.html', data=json.load(open(GRASP_JSON_PATH, 'r')))

@app.route('/pattern',methods = ['GET'])
def show_pattern():
	pid = int(request.args.get('pid'))
	data = json.load(open(GRASP_JSON_PATH, 'r'))
	max_ex = 100 
	if 'max' in request.args:
		max_ex = int(request.args.get('max'))

	pos_match = []
	for idx, v in enumerate(data['rules'][pid-1]['pos_example_labels']):
		if v:
			pos_match.append(data['dataset']['pos_exs'][idx])

	neg_match = []
	for idx, v in enumerate(data['rules'][pid-1]['neg_example_labels']):
		if v:
			neg_match.append(data['dataset']['neg_exs'][idx])

	more_pos = max_ex + 50 if max_ex < len(pos_match) else 0
	more_neg = max_ex + 50 if max_ex < len(neg_match) else 0

	return render_template('specific_pattern.html', 
		pid = pid, 
		pattern = data['rules'][pid-1],
		data_info = data['dataset']['info'],
		pos_match = pos_match[:max_ex][::-1],
		neg_match = neg_match[:max_ex][::-1],
		more_pos = more_pos,
		more_neg = more_neg)

@app.route('/examples',methods = ['GET'])
def show_examples():
	data = json.load(open(GRASP_JSON_PATH, 'r'))
	exs_per_page = 100 
	pos_page, neg_page = 1, 1
	if 'pos_page' in request.args:
		pos_page = int(request.args.get('pos_page'))
	if 'neg_page' in request.args:
		neg_page = int(request.args.get('neg_page'))

	first_pos = (pos_page - 1) * exs_per_page
	first_neg = (neg_page - 1) * exs_per_page
	max_pos_page = math.ceil(data['dataset']['info']['#pos'] / exs_per_page)
	max_neg_page = math.ceil(data['dataset']['info']['#neg'] / exs_per_page)

	return render_template('examples.html', 
		data_info = data['dataset']['info'],
		pos_match = data['dataset']['pos_exs'][first_pos:first_pos+exs_per_page],
		neg_match = data['dataset']['neg_exs'][first_neg:first_neg+exs_per_page],
		max_pos_page = max_pos_page,
		max_neg_page = max_neg_page,
		exs_per_page = exs_per_page,
		pos_page = pos_page,
		neg_page = neg_page,
		rules = data['rules'])

@app.route('/example',methods = ['GET'])
def show_example():
	data = json.load(open(GRASP_JSON_PATH, 'r'))
	the_class = request.args.get('class')
	eid = int(request.args.get('eid'))
	ex = data['dataset'][f'{the_class}_exs'][eid]

	match_rules = list(set([item for sublist in ex['rules'] for item in sublist]))
	match_rules.sort()
	
	pos_rules = []
	neg_rules = []
	for ridx in match_rules:
		r = data['rules'][ridx]
		rule_len = len(eval(r['pattern']))
		rule_meaning = r['meaning']
		rule_meaning = rule_meaning.replace(', closely followed by ', ', and then by ')
		rule_meaning = rule_meaning.replace(', immediately followed by ', ', and then by ')
		rule_fragments = rule_meaning.split(', and then by ')
		assert len(rule_fragments) == rule_len

		match_indices = [widx for widx, lst in enumerate(ex['rules']) if ridx in lst]

		ans = []
		color = []
		connector = 'immediately' if 'immediately' in r['meaning'] else 'closely'
		for sidx, f in enumerate(rule_fragments):
			if sidx == 0:
				ans.append(f)
				color.append(0)
			elif sidx == 1:
				ans.append(f', {connector} followed by {f}')
				color.append(0)
			else:
				ans.append(f', and then by {f}')
				color.append(0)
			ans.append(f' ({ex["tokens"][match_indices[sidx]]})')
			color.append(1)

		if r['#pos'] >= r['#neg']: # Pos
			pos_rules.append({
				'pid': ridx,
				'texts': ans,
				'color': color
				})
		else:
			neg_rules.append({
				'pid': ridx,
				'texts': ans,
				'color': color
				})
	return render_template('specific_example.html', 
		ex = ex,
		rules = data['rules'],
		pos_rules = pos_rules,
		neg_rules = neg_rules)

if __name__ == "__main__":
	app.run(debug=True)