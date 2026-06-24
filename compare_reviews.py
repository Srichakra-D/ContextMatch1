import json
old = {r['candidate_id'] for r in json.load(open('/home/sai/test/calibrations/reviews.json'))}
new = {r['candidate_id'] for r in json.load(open('/home/sai/test/calibration/reviews.json'))}
common = old & new
only_old = old - new
only_new = new - old
print('Common:', len(common))
print('Only in old:', len(only_old), list(only_old))
print('Only in new:', len(only_new), list(only_new))