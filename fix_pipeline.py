import sys
content = open('/home/sai/test/contextmatch/pipeline.py').read()
old = '            raise ValueError("comparison response must contain each group ID once")'
new = '            missing = [c for c in group if c not in result.ordered_candidate_ids]\n            if missing:\n                return result.ordered_candidate_ids + missing\n            raise ValueError("comparison response must contain each group ID once")'
assert old in content, 'Pattern not found!'
content = content.replace(old, new, 1)
open('/home/sai/test/contextmatch/pipeline.py', 'w').write(content)
print('Done')