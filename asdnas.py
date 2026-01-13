

map = {}
for i, num in enumerate(nums):
    complement = target - num
    if complement in map:
        map[num] = i